import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import numpy as np
from sklearn.cluster import KMeans

def initsetup(bulk,geneselection,batch):
    print('Start initial setup')
    bulkdata = anndata.read_h5ad(bulk)
    sids = list(bulkdata.obs['sample_ids'])
    sc.pp.normalize_total(bulkdata, target_sum=1e4)
    bulkdata.write('normed_bulkdata.h5ad')
    # write sample ids
    sids = []
    f = open('sids.txt','w')
    for sid in sids:
        f.write(sid+'\n')
    f.close()
    
    #preprocessing
    sc.pp.log1p(bulkdata)
    if geneselection == 'yes':
        sc.pp.highly_variable_genes(bulkdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    elif geneselection == 'no':
        pass
    else:
        sc.pp.highly_variable_genes(bulkdata, n_top_genes=6000)
    
    bulkdata = bulkdata[:, bulkdata.var.highly_variable]
    
    #record hvmask
    hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
    np.save('hvgenes.npy',hvgenes)
    
    #dim reduction and clustering
    sc.tl.pca(bulkdata, n_comps=100)
    #cluster
    BATCH_SIZE = batch
    kmeans = KMeans(n_clusters=BATCH_SIZE, random_state=0).fit(bulkdata.obsm['X_pca'])
    cluster_labels = kmeans.labels_
    #find representatives and cluster labels
    pnums = []
    for i in range(len(bulkdata.X)):
        pnums.append(i)
    pnums=np.array(pnums)
    centers=[]
    representatives=[]
    repredic={}
    for i in range(len(np.unique(cluster_labels))):
        mask = (cluster_labels==i)
        cluster = bulkdata.obsm['X_pca'][mask]
        cluster_patients = pnums[mask]
        center = cluster.mean(axis=0)
        centers.append(center)
        # find the closest patient
        sqdist = ((cluster - center)**2).sum(axis=1)
        cluster_representative = cluster_patients[np.argmin(sqdist)]
        representatives.append(cluster_representative)
        repredic[i] = cluster_representative
    centers = np.array(centers)
    #store representatives cluster labels
    if (os.path.isdir('status')) == False:
        os.sys('mkdir status')
    

    f=open('status/init_cluster_labels_4.txt','w')
    for i in range(len(cluster_labels)):
        f.write(str(cluster_labels[i])+'\n')
    f.close()

    f=open('status/init_representatives_4.txt','w')
    for i in range(len(representatives)):
        f.write(str(representatives[i])+'\n')
    f.close()
    
    print('Initial setup finished.')
    return




def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler initsetup")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--bulk',required=True,help="Input bulk data as a h5ad file. Sample IDs should be stored in obs.['sample_ids']. Gene symbols should be stored in var.index.")
    
    
    
    optional.add_argument('--geneselection',required=False,default='yes', help="Whether to perform highly variable gene selection: 'yes' or 'no'. (Default: yes)")
    
    optional.add_argument('--batch',required=False, default=4, help="The representative sample batch size (Default: 4)")
    
    args = parser.parse_args()
    bulk = args.bulk
    geneselection = args.geneselection
    batch = int(args.batch)
    
    initsetup(bulk,geneselection,batch)

if __name__=="__main__":
    main()
