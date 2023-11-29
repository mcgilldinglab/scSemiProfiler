import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import numpy as np
import faiss
import scipy

### evaluation functions

def faiss_knn(query, x, n_neighbors=1):
    n_samples = x.shape[0]
    n_features = x.shape[1]
    x = np.ascontiguousarray(x)
    
    index = faiss.IndexFlatL2(n_features)
    #index = faiss.IndexFlatIP(n_features)
                  
    index.add(x)
    
    if n_neighbors < 2:
        neighbors = 2
    else: 
        neighbors = n_neighbors
    
    weights, targets = index.search(query, neighbors)

    #sources = np.repeat(np.arange(n_samples), neighbors)
    #targets = targets.flatten()
    #weights = weights.flatten()
    weights = weights[:,:n_neighbors]
    if -1 in targets:
        raise InternalError("Not enough neighbors were found. Please consider "
                            "reducing the number of neighbors.")
    return weights

def pearson_compare(query,x):
    return 0

def cos_compare(query,x):
    return 0


def pca_compare(query,x):
    qx = np.concatenate([query,x],axis=0)
    qxpca = PCA(n_components=100)
    dx=qxpca.fit_transform(qx)
    
    newq = dx[:query.shape[0],:].copy(order='C')
    newx = dx[query.shape[0]:,:].copy(order='C')
    score = faiss_knn(newq,newx,n_neighbors=1)
    return score

def umap_compare(query,x):
    qx = np.concatenate([query,x],axis=0)
    qxpca = PCA(n_components=100)
    dpca=qxpca.fit_transform(qx)
    umap_reduc=umap.UMAP(min_dist=0.5,spread=1.0,negative_sample_rate=5 )
    dx = umap_reduc.fit_transform(dpca)
    newq = dx[:query.shape[0],:].copy(order='C')
    newx = dx[query.shape[0]:,:].copy(order='C')
    score = faiss_knn(newq,newx,n_neighbors=1)
    return score

def knncompare(query,x,n_neighbors=1,dist='PCA'):
    if dist == 'Euclidean':
        score = faiss_knn(query,x,n_neighbors=n_neighbors)
        score2 = faiss_knn(x,query,n_neighbors=n_neighbors)
    elif dist == 'Pearson':
        score = pearson_compare(query,x)
        score2 = pearson_compare(x,query)
    elif dist == 'cos':
        score = cos_compare(query,x)
        score2 = cos_compare(x,query)
    elif dist == 'PCA':
        score = pca_compare(query,x)
        score2 = pca_compare(x,query)
    elif dist == 'UMAP':
        score = umap_compare(query,x)
        score2 = umap_compare(x,query)
    else:
        score = 0
        print('distance option not found')
        
    return (score.mean() + score2.mean())/2

def normtotal(x,h=1e4):
    ratios = h/x.sum(axis=1)
    x=(x.T*ratios).T
    return x

## active learning functions 
def pick_batch(reduced_bulk=None,\
                representatives=None,\
                cluster_labels=None,\
                xdimsemis=None,\
                xdimgts=None,\
                discount_rate = 1,\
                semi_dis_rate = 1,\
                batch_size=8\
               ):
    # 
    lhet = []
    lmp = [] 
    for i in range(len(representatives)):
        cluster_heterogeneity,in_cluster_uncertainty,uncertain_patient=compute_cluster_heterogeneity(cluster_number=i,\
                            reduced_bulk=reduced_bulk,\
                           representatives=init_representatives,\
                            cluster_labels=init_cluster_labels,\
                            xdimsemis=xdimsemis,\
                            xdimgts=xdimgts,\
                            discount_rate = 1,\
                            semi_dis_rate = 1\
                           )
        lhet.append(cluster_heterogeneity)
        lmp.append(uncertain_patient)
    
    
    new_representatives = copy.deepcopy(representatives)
    for i in range(batch_size):
        mp_index = np.array(lhet).argmax()
        mp = lmp[mp_index]
        
        new_representatives.append(mp)
        lhet.pop(mp_index)
        lmp.pop(mp_index)
    
    new_cluster_labels= update_membership(reduced_bulk=reduced_bulk,\
                      representatives=new_representatives)
    
    return new_representatives,new_cluster_labels

def pick_batch_eee(reduced_bulk=None,\
                representatives=None,\
                cluster_labels=None,\
                xdimsemis=None,\
                xdimgts=None,\
                discount_rate = 1,\
                semi_dis_rate = 1,\
                batch_size=8\
               ):
    # 
    lhet = []
    lmp = [] 
    for i in range(len(representatives)):
        cluster_heterogeneity,in_cluster_uncertainty,uncertain_patient=compute_cluster_heterogeneity(cluster_number=i,\
                            reduced_bulk=reduced_bulk,\
                           representatives=representatives,\
                            cluster_labels=cluster_labels,\
                            xdimsemis=xdimsemis,\
                            xdimgts=xdimgts,\
                            discount_rate = 1,\
                            semi_dis_rate = 1\
                           )
        lhet.append(cluster_heterogeneity)
        lmp.append(uncertain_patient)
    
    new_representatives = copy.deepcopy(representatives)
    new_cluster_labels = copy.deepcopy(cluster_labels)
    print('heterogeneities: ',lhet)
    for i in range(batch_size):
        new_num = len(new_representatives)
        mp_index = np.array(lhet).argmax()
        print(mp_index)
        lhet[mp_index] = -999
        bestp, new_cluster_labels, hets = best_patient(cluster_labels=new_cluster_labels,representatives=new_representatives,\
                 reduced_bulk=reduced_bulk,cluster_num=mp_index,new_num=new_num)
        
        new_representatives = new_representatives + [bestp]
    
    return new_representatives,new_cluster_labels

def best_patient(cluster_labels=None,representatives=None,\
                 reduced_bulk=None,cluster_num=0,new_num=None):
    if new_num == None:
        new_num = len(representatives)
    pindices = np.where(np.array(cluster_labels)==cluster_num)[0]
    representative = representatives[cluster_num]
    hets=[]
    potential_new_labels = []
    for i in range(len(pindices)):
        potential_new_label = copy.deepcopy(cluster_labels)
        newrepre = pindices[i]
        het = 0
        if newrepre in representatives:
            hets.append(9999)
            potential_new_labels.append(potential_new_label)
            continue
        for j in range(len(pindices)):
            brepre = reduced_bulk[representative]
            brepre2 = reduced_bulk[newrepre]
            bj = reduced_bulk[pindices[j]]
            bdist1 = (brepre - bj)**2
            bdist1 = bdist1.sum()
            bdist1 = bdist1**0.5
            bdist2 = (brepre2 - bj)**2
            bdist2 = bdist2.sum()
            bdist2 = bdist2**0.5
            
            if bdist1 > bdist2:
                #print(pindices[j])
                het = het + bdist2
                potential_new_label[pindices[j]]=new_num
            else:
                het = het + bdist1
        hets.append(het)
        potential_new_labels.append(potential_new_label)
    hets = np.array(hets)
    bestp = pindices[np.argmin(hets)]
    new_cluster_labels = potential_new_labels[np.argmin(hets)]
    return bestp, new_cluster_labels, hets

def update_membership(reduced_bulk=None,\
                      representatives=None,\
                      
                     ):
    new_cluster_labels = []
    for i in range(len(reduced_bulk)):
        
        dists=[]
        #dist to repres
        for j in representatives:
            bdist = (reduced_bulk[j] - reduced_bulk[i])**2 
            bdist = bdist.sum()
            bdist = bdist**0.5
            dists.append(bdist)
        membership = np.array(dists).argmin()
        new_cluster_labels.append(membership)
    return new_cluster_labels

def compute_cluster_heterogeneity(cluster_number=0,\
                            reduced_bulk=None,\
                           representatives=None,\
                            cluster_labels=None,\
                            xdimsemis=None,\
                            xdimgts=None,\
                            discount_rate = 1,\
                            semi_dis_rate = 1\
                           ):
    semiflag=0
    representative = representatives[cluster_number]
    in_cluster_uncertainty = []
    cluster_labels = np.array(cluster_labels)
    cluster_patient_indices = np.where(cluster_labels==cluster_number)[0]
    
    for i in range(len(cluster_patient_indices)): # number of patients in this cluster except the representative
        
        patient_index = cluster_patient_indices[i]
        
        if patient_index in representatives:
            in_cluster_uncertainty.append(0)
            continue
            
        # distance between this patient and representative
        bdist = (reduced_bulk[representative] - reduced_bulk[patient_index])**2 
        bdist = bdist.sum()
        bdist = bdist**0.5
        
        ma = np.array(xdimsemis[patient_index]).copy(order='C')
        mb = np.array(xdimgts[representative]).copy(order='C')
        sdist = (faiss_knn(ma,mb,n_neighbors=1).mean())
        
        semiloss = np.log(1+gts[patient_index].sum(axis=0))- np.log(1+semis[patient_index].sum(axis=0))
        semiloss = semiloss**2
        semiloss = semiloss.sum()
        semiloss = semiloss**0.5
        
        uncertainty = bdist + sdist*discount_rate + semi_dis_rate * semiloss
        
        in_cluster_uncertainty.append(uncertainty)
        
    cluster_heterogeneity = np.array(in_cluster_uncertainty).sum()
    uncertain_patient = cluster_patient_indices[np.array(in_cluster_uncertainty).argmax()] 

    return cluster_heterogeneity,in_cluster_uncertainty,uncertain_patient



def activeselection(representatives,cluster,lambdasc,lambdapb):

    rep = []
    f = open(representatives,'r')
    lines = f.readlines()
    for l in lines:
        rep.append(l)
    f.close()
    
    cl=[]
    f = open(cluster,'r')
    lines = f.readlines()
    for l in lines:
        cl.append(l)
    f.close()
    
    bulkdata = anndata.read_h5ad('processed_bulkdata.h5ad')
    reduced_bulk = bulkdata.obsm['X_pca']
    
    #acquire semi-profiled cohort

    hvgenes = np.load('hvgenes.npy')
    
    adata = anndata.read_h5ad('sample_sc/' + rep[0] + '.h5ad')
    hvmask = []
    for g in adata.var.index:
        if g in hvgenes:
            hvmask.append(True)
        else:
            hvmask.append(False)
    hvmask = np.array(hvmask)
    
    xsemi = []
    for i in range(len(sids)):
        sid = sids[i]
        representative = rep[cl[i]]
        xsemi.append(np.load('inferreddata/'+sids[representative]+'to'+sid+'.npy'))
        print(i,end=', ')

    
    
    nrep, nlabels = pick_batch_eee(reduced_bulk = reduced_bulk,\
                    representatives = rep,\
                    cluster_labels = cl,\
                    xdimsemis=xsemi,\
                    xdimgts=xsemi,\
                    discount_rate = lambdasc,\
                    semi_dis_rate = lambdapb,\
                    batch_size=4\
                   )
    new_representatives = nrep
    new_cluster_labels = nlabels
    f=open('status/eer_cluster_labels_'+str(rnd+1)+'.txt','w')
    for i in range(len(new_cluster_labels)):
        f.write(str(new_cluster_labels[i])+'\n')
    f.close()
    f=open('status/eer_representatives_'+str(rnd+1)+'.txt','w')
    for i in range(len(new_representatives)):
        f.write(str(new_representatives[i])+'\n')
    f.close()


    return




def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler initsetup")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--representatives',required=True,help="A txt file including all the IDs of the representatives used in the current round of semi-profiling.")
    
    required.add_argument('--cluster',required=True,help="A txt file specifying the cluster membership.")
    
    optional.add_argument('--lambdasc',required=False,default='1.0', help="Scaling factor for the single-cell transformation difficulty from the representative to the target (Default: 1.0)")
    
    optional.add_argument('--lambdapb',required=False, default='1.0', help="Scaling factor for the pseudobulk data difference (Default: 1.0)")
    
    args = parser.parse_args()
    representatives = args.representatives
    cluster = args.cluster
    lambdasc = float(args.lambdasc)
    lambdapb = float(args.lambdapb)
    activeselection(representatives,cluster,lambdasc,lambdapb)

if __name__=="__main__":
    main()
