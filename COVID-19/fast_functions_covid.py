import numpy as np
import torch
import os
import timeit
import copy

import anndata
from anndata import AnnData
import scanpy as sc
from sklearn.neighbors import kneighbors_graph

from fast_generator_covid import *


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



def setdata(pid,tgtpid=None,device='cuda:5',k=15,diagw=1.0,bulksource = -1):
    #adata = anndata.read_h5ad('COVID_HV_GT/'+pid+'.h5ad')
    adata = anndata.read_h5ad('sample_sc/'+pid+'.h5ad') ###
    sc.pp.normalize_total(adata,1e4)
    adata.X = adata.X.todense()
    hvmask = np.load('hvmask.npy')
    adata = adata[:,hvmask]
    #(0) basic processing
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #adata.X = np.exp(adata.X.todense())-1
    #hvg = np.load('new_hvg.npy')
    #adata = adata[:,hvg]
    adata.obs['cellidx']=range(len(adata.obs))
    #adata.X = adata.X.todense()
    
    #(1) cell graph
    adata,adj = fast_cellgraph(adata,k,diagw)
    
    #(2) geneset
    adata,geneset_len = get_geneset(adata,pid)
    
    #(3) variances
    X=torch.tensor(adata.X)
    variances = (X.var(dim=0)).to(device)
    #variances = None
    
    #(4) 
    hvmask = np.load('hvmask.npy')
    pseudobulk = np.array(adata.X.mean(axis=0)).reshape((-1))
    
    if bulksource == 0: # pid
        bulknorm = anndata.read_h5ad('bulkdata.h5ad')
        bulknorm = bulknorm[:,hvmask]
        pidnum = list(bulknorm.obs['pids']).index(pid)
        bulks = bulknorm.X
        bulk = np.array(bulks[pidnum]).reshape((-1))
        pseudobulk[:hvmask.sum()]=bulk
        bulk = pseudobulk
    elif bulksource == 1: # tgtpid
        bulknorm = anndata.read_h5ad('bulkdata.h5ad')
        bulknorm = bulknorm[:,hvmask]
        tgtpidnum = list(bulknorm.obs['pids']).index(tgtpid)
        bulks = bulknorm.X
        bulk = np.array(bulks[tgtpidnum]).reshape((-1))
        pseudobulk[:hvmask.sum()]=bulk
        bulk = pseudobulk
    elif bulksource == -1:
        bulk = None
    
    fastgenerator.setup_anndata(adata)

    return adata,adj,variances,bulk,geneset_len


def get_geneset(adata,pid):
    #geneset = np.load('COVID_geneset/'+pid+'.npy')
    geneset = np.load('geneset/'+pid+'.npy')
    setmask = np.load('hvset.npy')
    geneset = geneset[:,setmask]
    
    features = np.concatenate([adata.X,geneset],1)
    bdata = AnnData(features,dtype='float32')
    bdata.obs = adata.obs
    bdata.obsm = adata.obsm
    geneset_len = geneset.shape[1]
    adata = bdata.copy()
    
    #if 'neighborx' in adata.obsm.keys():
    #    adata.obsm['neighborx'] = np.concatenate([adata.obsm['neighborx'],adata.X[:,2000:]],axis=1)
    
    return adata,geneset_len



def fastrecon(pid,tgtpid=None,device='cuda:5',k=15,diagw=1.0,vaesteps=150,gansteps=150,lr=1e-3,save=True,path=None):
    
    #set data
    adata,adj,variances,bulk,geneset_len = setdata(pid,tgtpid,device,k,diagw,bulksource=-1)
    
    # train
    model = fastgenerator(adj,variances,None,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
    steps=3
    model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*0.001},use_gpu=device)
    for i in range(gansteps):
        print(i,end=', ')
        model.train(max_epochs=steps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*0.001},use_gpu=device)
        model.train(max_epochs=steps, plan_kwargs={'lr':1e-10,'lr2':lr,'kappa':4040*0.001},use_gpu=device)
    
    # save model
    if save == True:
        if path == None:
            path = 'covid_models/fast_reconst1_'+pid
        torch.save(model.module.state_dict(), path)
    
    return model



# reconst stage 2
def reconst_pretrain2(pid,premodel,device='cuda:5',k=15,diagw=1.0,vaesteps=50,gansteps=50,lr=1e-4,save=True,path=None):
    adata,adj,variances,bulk,geneset_len = setdata(pid,None,device,k,diagw,bulksource=0)
    
    #(4) bulk
    bulk = (np.array(adata.X)).mean(axis=0)
    bulk = bulk.reshape((-1))
    bulk = torch.tensor(bulk).to(device)
    
    #(5) reconstruct pretrain
    fastgenerator.setup_anndata(adata)
    model = fastgenerator(adj = adj,variances = variances,markermask = None,bulk=bulk,geneset_len = geneset_len,adata=adata,\
                n_hidden=256,n_latent=32,dropout_rate=0,countbulkweight=1,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,\
                power=2,corrbulkweight=0,meanbias=0)
    
    if type(premodel) == type(None):
        1+1
        #model.module.load_state_dict(torch.load('saved_models/fastreconst2_'+pid))
    else:
        model.module.load_state_dict(premodel.module.state_dict())
    
    steps=3
    model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*0.001},use_gpu=device)
    for i in range(gansteps):
        print(i,end=', ')
        model.train(max_epochs=steps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*0.001},use_gpu=device)
        model.train(max_epochs=steps, plan_kwargs={'lr':1e-10,'lr2':lr,'kappa':4040*0.001},use_gpu=device)
    
    if save == True:
        if path == None:
            path = 'covid_models/fastreconst2_'+pid
        torch.save(model.module.state_dict(), path)
        
    return model





    
    
    