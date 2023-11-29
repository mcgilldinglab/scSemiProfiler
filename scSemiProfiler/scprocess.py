import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import torch
import numpy as np
import gc
import pandas as pd
import timeit
import warnings
warnings.filterwarnings('ignore')
import faiss
from sklearn.cluster import KMeans
import sklearn
from scipy import stats
from sklearn.neighbors import kneighbors_graph
#from datasets import AnnDataset, NumpyDataset
from matplotlib.pyplot import figure
#from fast_generator import *
#from fast_functions import *

from torch.utils.data import Dataset
import anndata

class AnnDataset(Dataset):
    def __init__(self, filepath: str, label_name: str = None, second_filepath: str = None,
                 variable_gene_name: str = None):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = sc.read(filepath, dtype='float64', backed="r")

        #genes = self.data.var.index.values
        if 'genes' in self.data.var:
            genes = self.data.var['genes']
        else:
            genes = self.data.var.index 
        self.genes_upper = [g.upper() for g in genes]
        if label_name is not None:
            self.clusters_true = self.data.obs[label_name].values
        elif 'celltype' in self.data.obs:
            self.cluusters_true = self.data.obs['celltype']
        else:
            self.clusters_true = None

        self.N = self.data.shape[0]
        self.G = len(self.genes_upper)

        self.secondary_data = None
        if second_filepath is not None:
            self.secondary_data = np.load(second_filepath)
            assert len(self.secondary_data) == self.N, "The other file have same length as the main"

        if variable_gene_name is not None:
            #_idx = np.where(self.data.var[variable_gene_name].values)[0]
            #self.exp_variable_genes = self.data.X[:, _idx]
            #self.variable_genes_names = self.data.var.index.values[_idx]
            
            
            # jt's version
            vgmask=[]
            for g in genes:
                vgmask.append(g in variable_gene_name)
            vgmask=np.array(vgmask)
            self.exp_variable_genes = self.data.X[:, vgmask]
            self.variable_genes_names = variable_gene_name
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        main = self.data[idx].X.flatten()
       # main = sc.pp.log1p(main)
        if self.secondary_data is not None:
            secondary = self.secondary_data[idx].flatten()
            return main, secondary
        else:
            return main
        
def fast_cellgraph(adata,k,diagw):
    adj = kneighbors_graph(np.array(adata.X), k, mode='connectivity', include_self=True)
    adj = adj.toarray()
    diag = np.array(np.identity(adj.shape[0]).astype('float32'))*diagw
    adj = adj + diag
    adj = adj/adj.sum(axis=1)        
    selfw = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        selfw[i] = adj[i,i]
    selfw=selfw.astype('float32')
    adata.obs['selfw']=selfw
    #remove self so that not in neighbors
    for i in range(adj.shape[0]):
        adj[i,i]=0
    adj = torch.from_numpy(adj.astype('float32'))#.type(torch.FloatTensor)
    neighboridx = np.where(adj!=0)
    xs = neighboridx[0]
    ys = neighboridx[1]
        
    maxn=k
    neighbors = np.zeros((adj.shape[0],maxn-1)) - 1
    for i in range(len(adata.obs)):
        ns=np.zeros(maxn-1)-1
        flag=0
        j=0
        k=0
        while flag!=2 and j<xs.shape[0]:
            if xs[j]==i:
                ns[k] = (ys[j])
                k+=1
                flag=1
            elif flag==1:
                flag=2
            j+=1
        neighbors[i] = ns
        
    neighbors = neighbors.astype(int)
    adata.obsm['neighbors']=neighbors
    neighborx = np.array(adata.X)
    
    normchoice = 0
    if normchoice == 0:
        neighborx = np.log(1+neighborx)*selfw[0] + np.log(1+neighborx[neighbors,:]).sum(axis=1)*(1-selfw[0])/(maxn-1)
    else:                                       ## 2*
        neighborx = neighborx*selfw[0] + neighborx[neighbors].sum(axis=1)*(1-selfw[0])/(maxn-1)
        neighborx = np.log(1+neighborx)
    
    adata.obsm['neighborx']=neighborx
    
    return adata,adj


        
def scprocess(singlecell,cellfilter,threshold,geneset,weight,k):
    
    print('Processing representative single-cell data')
    
    scdata = anndata.read_h5ad(singlecell)
    sids = np.unique(scdata.obs['sample_ids'])
    
    print('Filtering cells')
    # cell filtering
    if cellfilter == 'yes':
        sc.pp.filter_cells(scdata, min_genes=200)
        scdata.var['mt'] = scdata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(scdata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        scdata = scdata[scdata.obs.n_genes_by_counts < 2500, :]
        scdata = scdata[scdata.obs.pct_counts_mt < 5, :]
    
    print('Removing background noise')
    # norm remove noise
    sc.pp.normalize_total(scdata, target_sum=1e4)        
    if float(threshold) > 0:
        X = scdata.X
        cutoff = 1e4*threshold
        X = X*[X>cutoff]
        nscdata = anndata.AnnData(X)
        nscdata.obs = scdata.obs
        nscdata.var = scdata.var
        nscdata.uns = scdata.uns
        scdata = nscdata
    
    
    
    # store singlecell data, geneset score
    if (os.path.isdir('sample_sc')) == False:
        os.sys('mkdir sample_sc')
    if (os.path.isdir('geneset_scores')) == False:
        os.sys('mkdir geneset_scores')
        
    if geneset != 'none':
        prior_name = "c2.cp.v7.4.symbols.gmt" # "c5.go.bp.v7.4.symbols.gmt+c2.cp.v7.4.symbols.gmt+TF-DNA"
    

        
    print('Computing geneset scores')
    zps=[]
    for sid in sids:
        adata = scdata[scdata.obs['sample_ids'] == 'sid']
        X = adata.X

        gene_sets_path = "genesets/"
        expression_only = AnnDataset(data_filepath, label_name=label_name, variable_gene_name=variable_gene_name)
        exp_variable_genes = expression_only.exp_variable_genes
        variable_genes_names = expression_only.variable_genes_names
        genes_upper = expression_only.genes_upper
        clusters_true = expression_only.clusters_true
        N = expression_only.N
        G = expression_only.G
        gene_set_matrix, keys_all = getGeneSetMatrix(prior_name, genes_upper, gene_sets_path)
        
        zp = X.dot(np.array(gene_set_matrix).T)
        eps = 1e-6
        den = (np.array(gene_set_matrix.sum(axis=1))+eps)
        zp = (zp+eps)/den
        zp = zp - eps/den
        np.save('geneset_scores/'+pid,zp)
        zps.append(zp)
    
    if 'hvset.npy' not in os.listdir():
        zps=np.array(zps)
        zdata = anndata.AnnData(zps)
        sc.pp.log1p(zdata)
        sc.pp.highly_variable_genes(zdata)
        hvset = zdata.var.highly_variable
        np.save('hvset.npy',hvset)

    
        
    # select highly variable genes
    hvgenes = np.load('hvgenes.npy')
    hvmask = []
    for i in scdata.var.index:
        if i in hvgenes:
            hvmask.append(True)
        else:
            hvmask.append(False)
    hvmask = np.array(hvmask)
    scdata = scdata[:,hvmask]
    
    

    for sid in sids:
        adata = scdata[scdata.obs['sample_ids'] == 'sid']
        
        #gcn
        adata.obs['cellidx']=range(len(adata.obs))
        adata,adj = fast_cellgraph(adata,k,diagw)

        variances = (adata.X.var(dim=0))
        
        adata.write('sample_sc/' + sid + '.h5ad')
    
    print('Finished processing representative single-cell data')
    return 




def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler scprocess")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--singlecell',required=True,help="Input representatives' single-cell data as a h5ad file. Sample IDs should be stored in obs.['sample_ids']. Cell IDs should be stored in obs.index. Gene symbols should be stored in var.index. Values should either be raw read counts or normalized expression.")
    
    
    optional.add_argument('--cellfilter',required=False, default='yes', help="Whether to perform cell filtering: 'yes' or 'no'. (Default: yes)")
    optional.add_argument('--threshold',required=False, default='1e-3', help="The threshold for removing extremely low expressed background noise, as a proportion of the library size. (Default: 1e-3)")
    optional.add_argument('--geneset',required=False, default='human', help="Specify the gene set file: 'human', 'mouse', 'none', or path to the file (Default: 'human')")
    optional.add_argument('--weight',required=False, default=0.5, help="The proportion of top highly variable features to increase importance weight. (Default: 0.5)")
    optional.add_argument('--k',required=False, default=15, help="K-nearest cell neighbors used for cell graph convolution. (Default: 15)")
    
    args = parser.parse_args()
    singlecell = args.singlecell
    cellfilter = args.cellfilter
    threshold = args.threshold
    geneset = args.geneset
    weight = args.weight
    k = args.k
    
    scprocess(singlecell,cellfilter,threshold,geneset,weight,k)

if __name__=="__main__":
    main()
