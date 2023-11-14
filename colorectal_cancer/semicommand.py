import torch
import umap
import anndata
import numpy as np
import os
from os import sys
import gc
import pandas as pd
import timeit
import warnings
warnings.filterwarnings('ignore')
import faiss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn
from scipy import stats
import scanpy as sc
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph
#from datasets import AnnDataset, NumpyDataset
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cos
import anndata
from matplotlib.pyplot import figure
import copy


from fast_generator_cancer import *
from fast_functions_cancer import *


pp = 8
dis=3
def unisemi0(adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model0 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight =1*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                      power=2,upperbound=99999,meanbias=0)
    if type(premodel)==type('string'):
        model0.module.load_state_dict(torch.load(premodel))
    else:
        model0.module.load_state_dict(premodel.module.state_dict())
    
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4
    model0.train(max_epochs=400//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model0.module.state_dict(), 'tmp/model0')
    return model0.history


def unisemi1(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model1 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 4*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model1.module.load_state_dict(torch.load('tmp/model0'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4
    model1.train(max_epochs=400//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model1.module.state_dict(), 'tmp/model1')
    return model1.history

def unisemi2(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model2 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 16*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model2.module.load_state_dict(torch.load('tmp/model1'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4
    model2.train(max_epochs=200//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model2.module.state_dict(), 'tmp/model2')
    return model2.history

def unisemi3(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model3 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 64*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model3.module.load_state_dict(torch.load('tmp/model2'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4
    model3.train(max_epochs=200//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model3.module.state_dict(), 'tmp/model3')
    return model3.history

def unisemi4(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model4 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 128*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model4.module.load_state_dict(torch.load('tmp/model3'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4
    model4.train(max_epochs=400//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model4.module.state_dict(), 'tmp/model4')
    return model4.history

def unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model.module.load_state_dict(torch.load('tmp/model4'))
    lr = adata.X.shape[0] / (4e3) * 4e-4
    lr = 2e-4
    model.train(max_epochs=200//dis, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model.module.state_dict(), 'tmp/model')
    return model.history

def fast_semi(reprepid,tgtpid,premodel,device='cuda:1',k=15,diagw=1.0):
    adata,adj,variances,bulk,geneset_len = setdata(reprepid,tgtpid,device,k,diagw)
    #adata.X = (adata.X.todense())
    
    maxexpr = adata.X[:,:4053].max()
    upperbounds = [maxexpr/2, maxexpr/4, maxexpr/8, maxexpr/(8*np.sqrt(2)),maxexpr/16, maxexpr/32,maxexpr/64]     
       
    hvmask = np.load('hvmask.npy')
    #(5) tgt bulk
    #tgtdata = anndata.read_h5ad('COVID_HV_GT/'+tgtpid+'.h5ad')
    tgtdata = anndata.read_h5ad('sample_sc/'+tgtpid+'.h5ad')
    tgtdata.X = tgtdata.X.todense()
    
    
    tgtbulk = np.array(tgtdata.X)[:,hvmask]
    tgtbulk = tgtbulk.mean(axis=0)
    tgtbulk = np.array(tgtbulk).reshape((1,-1))
    bulk = adata.X.mean(axis=0)
    bulk = np.array(bulk).reshape((1,-1))
    bulk[:,:tgtbulk.shape[1]] = tgtbulk
    bulk = torch.tensor(bulk).to(device)
    
    batch_size=np.min([adata.X.shape[0],4500])
    
    
    #(6) semiprofiling
    fastgenerator.setup_anndata(adata)
    histdic={}
    
    hist = unisemi0(adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)

    histdic['total0'] = hist['train_loss_epoch']
    histdic['bulk0'] = hist['kl_global_train']

    gc.collect()
    torch.cuda.empty_cache() 

    hist = unisemi1(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[0],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)

    histdic['total1'] = hist['train_loss_epoch']
    histdic['bulk1'] = hist['kl_global_train']

    gc.collect()
    torch.cuda.empty_cache() 
    hist = unisemi2(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[1],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)

    histdic['total2'] = hist['train_loss_epoch']
    histdic['bulk2'] = hist['kl_global_train']
    #del model1
    gc.collect()

    #time.sleep(10)
    torch.cuda.empty_cache() 
    
    hist = unisemi3(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[2],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
    histdic['total3'] = hist['train_loss_epoch']
    histdic['bulk3'] = hist['kl_global_train']
    
    
    #del model2
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    hist = unisemi4(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[3],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)

    histdic['total4'] = hist['train_loss_epoch']
    histdic['bulk4'] = hist['kl_global_train']
    
    #del model3
    gc.collect()
    torch.cuda.empty_cache() 
 
    #unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[4],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
    #histdic['total'] = model.history['train_loss_epoch']
    #histdic['bulk'] = model.history['kl_global_train']
    
    gc.collect()
    torch.cuda.empty_cache() 

    model = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                     markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbounds[3],meanbias=0)
    model.module.load_state_dict(torch.load('tmp/model4'))
    # reconstruction
    xsemi = []
    scdl = model._make_data_loader(
            adata=adata,batch_size=batch_size
    )

    for tensors in scdl:
        samples = model.module.sample(tensors, n_samples=1)
        #samples = modelf.module.nb_sample(tensors, n_samples=1)
        xsemi.append(samples)
    xsemi = np.array(torch.cat(xsemi))[:,:4053]
    
    torch.save(model.module.state_dict(), 'cancer_models/semi_'+reprepid+"_to_"+tgtpid)
    np.save('semidata/fast'+ reprepid+'_to_'+tgtpid,xsemi)
    
    del model
    gc.collect()
    torch.cuda.empty_cache() 
    return histdic,xsemi




###  READ CONSTANTS

pids=[]
f = open('sids.txt','r')
lines=f.readlines()
for l in lines:
    pids.append(l.strip())

    
    
### READ VARIABLES

i = int(sys.argv[1])
device = sys.argv[2]
rnd = int(sys.argv[3])

reprefile = 'training_rec/eer_representatives_' + str(rnd) + '.txt' 
clusterfile = 'training_rec/eer_cluster_labels_' + str(rnd) + '.txt' 

f= open(reprefile,'r')
lines=f.readlines()
representatives=[]
for l in lines:
    representatives.append(int(l.strip().split()[0]))
f.close()

f= open(clusterfile,'r')
cluster_labels=[]
lines=f.readlines()
for l in lines:
    cluster_labels.append(int(l.strip().split()[0]))
f.close()


##### 

#print(cluster_labels)




### run iter




ta=timeit.default_timer()

if i not in representatives:
    tgtpid = pids[i]
    cluster = cluster_labels[i]
    repre_num = representatives[cluster]
    reprepid = pids[repre_num]
    premodel = 'cancer_models/fastreconst2_'+reprepid
    
    hists,xsemi = fast_semi(reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
    np.save('training_hist/hist_'+tgtpid,np.array(hists))
    torch.cuda.empty_cache() 
    tb=timeit.default_timer()
    print(str(tb-ta),'sec for',i)



