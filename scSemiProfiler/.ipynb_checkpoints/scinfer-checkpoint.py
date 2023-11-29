import numpy as np
import torch
import os
import timeit
import copy

import anndata
from anndata import AnnData
import scanpy as sc
from sklearn.neighbors import kneighbors_graph

from fast_generator import *


def setdata(pid,tgtpid=None,device='cuda:0',k=15,diagw=1.0):
    #adata = anndata.read_h5ad('COVID_HV_GT/'+pid+'.h5ad')
    adata = anndata.read_h5ad('sample_sc/'+pid+'.h5ad') ###
    
    #(2) geneset
    adata,geneset_len = get_geneset(adata,pid)
    
    #(3) variances
    X=torch.tensor(adata.X)
    variances = (X.var(dim=0)).to(device)
    
    #(4) bulk
    bulk = np.array(adata.X.mean(axis=0)).reshape((-1))
    fastgenerator.setup_anndata(adata)

    return adata,adj,variances,bulk,geneset_len


def get_geneset(adata,pid):
    #geneset = np.load('COVID_geneset/'+pid+'.npy')
    geneset = np.load('geneset_scores/'+pid+'.npy')
    setmask = np.load('hvset.npy')
    geneset = geneset[:,setmask]
    
    features = np.concatenate([adata.X,geneset],1)
    bdata = AnnData(features,dtype='float32')
    bdata.obs = adata.obs
    bdata.obsm = adata.obsm
    geneset_len = geneset.shape[1]
    adata = bdata.copy()
    
    return adata,geneset_len

def fastrecon(pid,tgtpid=None,device='cuda:0',k=15,diagw=1.0,vaesteps=100,gansteps=100,lr=1e-3,save=True,path=None):
    
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
            if (os.path.isdir('models')) == False:
                os.sys('mkdir models')
            path = 'models/fast_reconst1_'+pid
        torch.save(model.module.state_dict(), path)
    
    return model

# reconst stage 2
def reconst_pretrain2(pid,premodel,device='cuda:0',k=15,diagw=1.0,vaesteps=50,gansteps=50,lr=1e-4,save=True,path=None):
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
            path = 'models/fastreconst2_'+pid
        torch.save(model.module.state_dict(), path)
        
    return model


pp=8

dis = 1 #0.2

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
    lr = 2e-4 * dis
    model0.train(max_epochs=400, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model0.module.state_dict(), 'tmp/model0')
    return model0.history


def unisemi1(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model1 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 4*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model1.module.load_state_dict(torch.load('tmp/model0'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4 * dis
    model1.train(max_epochs=400, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model1.module.state_dict(), 'tmp/model1')
    return model1.history

def unisemi2(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model2 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 16*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model2.module.load_state_dict(torch.load('tmp/model1'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4 * dis
    model2.train(max_epochs=200, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model2.module.state_dict(), 'tmp/model2')
    return model2.history

def unisemi3(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model3 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 64*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model3.module.load_state_dict(torch.load('tmp/model2'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4 * dis
    model3.train(max_epochs=200, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model3.module.state_dict(), 'tmp/model3')
    return model3.history

def unisemi4(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model4 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 128*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model4.module.load_state_dict(torch.load('tmp/model3'))
    lr = adata.X.shape[0] / (4e3) * 2e-4
    lr = 2e-4 * dis
    model4.train(max_epochs=400, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model4.module.state_dict(), 'tmp/model4')
    return model4.history

def unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512*pp,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model.module.load_state_dict(torch.load('tmp/model4'))
    lr = adata.X.shape[0] / (4e3) * 4e-4
    lr = 2e-4 * dis
    model.train(max_epochs=200, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model.module.state_dict(), 'tmp/model')
    return model.history

def fast_semi(reprepid,tgtpid,premodel,device='cuda:0',k=15,diagw=1.0):
    adata,adj,variances,bulk,geneset_len = setdata(reprepid,tgtpid,device,k,diagw)
    #adata.X = (adata.X.todense())
    
    varainces = None
    
    maxexpr = adata.X.max()
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
    
    batch_size=np.min([adata.X.shape[0],60000])
    
    
    #(6) semiprofiling
    fastgenerator.setup_anndata(adata)

    
    hist = unisemi0(adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)
    histdic={}
    histdic['total0'] = hist['train_loss_epoch']
    histdic['bulk0'] = hist['kl_global_train']
    #del premodel
    gc.collect()
    torch.cuda.empty_cache() 
    #import time
    #time.sleep(10)
    hist = unisemi1(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[0],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total1'] = hist['train_loss_epoch']
    histdic['bulk1'] = hist['kl_global_train']
    #del model0
    gc.collect()
    torch.cuda.empty_cache() 
    hist = unisemi2(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[1],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total2'] = hist['train_loss_epoch']
    histdic['bulk2'] = hist['kl_global_train']
    #del model1
    gc.collect()

    #time.sleep(10)
    torch.cuda.empty_cache() 
    
    hist = unisemi3(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[2],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total3'] = hist['train_loss_epoch']
    histdic['bulk3'] = hist['kl_global_train']
    
    
    #del model2
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    hist = unisemi4(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[3],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total4'] = hist['train_loss_epoch']
    histdic['bulk4'] = hist['kl_global_train']
    
    #del model3
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    #hist = unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[4],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
    #histdic['total'] = hist['train_loss_epoch']
    #histdic['bulk'] = hist['kl_global_train']


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
        xsemi.append(samples)
    
    gc.collect()
    torch.cuda.empty_cache() 
    return histdic,xsemi,model


def scinfer(representatives,cluster,targetid,bulktype,lambdad,pretrain1batch,pretrain1lr,pretrain1vae,pretrain1gan,lambdabulkr,pretrain2lr, pretrain2vae,pretrain2gan,inferepochs,lambdabulkt,inferlr):
    
    if (os.path.isdir('inferreddata')) == False:
        os.sys('mkdir inferreddata')
        
    if representatives[-3:] == 'txt':
        print('Start single-cell inference in cohort mode')
        
        
        sids = []
        f = open('sids.txt','r')
        lines = f.readlines()
        for l in lines:
            sids.append(l.strip())
        f.close()
        
        
        repres = []
        f=open(representatives,'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            repres.append(l)
        
        cluster_labels = []
        f=open(cluster,'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            cluster_labels.append(l)
        
        
        print('pretrain1: reconstruction')
        repremodels = []
        c=0
        for rp in repres:
            device = 'cuda:0'
            sid = sids[rp]
            repremodels.append(\
                               fastrecon(pid=sid,\
                              tgtpid=None,device=device,k=15,diagw=1,vaesteps=int(pretrain1vae),gansteps=int(pretrain1gan),save=True,path=None)\
                              )
            
        print('pretrain2: reconstruction with representative bulk loss')
        repremodels2=[]
        for rp in representatives:
            sid = sids[rp]
            repremodels2.append(reconst_pretrain2(sid,repremodels[i],device,k=15,diagw=1.0,vaesteps=int(pretrain2vae),gansteps=int(pretrain2gan),save=True))
        
        print('inference')
        for i in range(len(sids)):
            if i not in representatives:
                tgtpid = sids[i]
                reprepid = sids[repres[cluster_labels[i]]]
                fast_semi(reprepid,tgtpid,premodel,device='cuda:0',k=15,diagw=1.0)
        
    else:
        print('Start single-cell inference in single-sample mode')
        
        repre = representatives
        target = targetid
        print('pretrain1: reconstruction')
        repremodel = fastrecon(pid=representatives, tgtpid=None,device=device,k=15,diagw=1,vaesteps=int(pretrain1vae),gansteps=int(pretrain1gan),save=True,path=None)
        print('pretrain2: reconstruction with representative bulk loss')
        premodel = reconst_pretrain2(repre,repremodel,device,k=15,diagw=1.0,vaesteps=int(pretrain2vae),gansteps=int(pretrain2gan),save=True)
        print('inference')
        fast_semi(repre,target,premodel,device='cuda:0',k=15,diagw=1.0)
    
    print('Finished single-cell inference')
    return




def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler scinfer")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--representatives',required=True,help="Either a txt file including all the IDs of the representatives used in the current round of semi-profiling when running in cohort mode, or a single sample ID when running in single-sample mode.")
    
    optional.add_argument('--cluster',required=False,default='na', help="A txt file specifying the cluster membership. Required when running in cohort mode.")
    
    optional.add_argument('--targetid',required=False, default='na', help="Sample ID of the target sample when running in single-sample mode.")
    
    optional.add_argument('--bulktype',required=False, default='real', help="Specify 'pseudo' for pseudobulk or 'real' for real bulk data. (Default: real)")
    
    optional.add_argument('--lambdad',required=False, default='4.0', help="Scaling factor for the discriminator loss for training the VAE generator. (Default: 4.0)")
    
    optional.add_argument('--pretrain1batch',required=False, default='128', help="Sample Batch Size of the first pretrain stage. (Default: 128)")
    
    optional.add_argument('--pretrain1lr',required=False, default='1e-3', help="Learning rate of the first pretrain stage. (Default: 1e-3)")
    
    optional.add_argument('--pretrain1vae',required=False, default='100', help = "The number of epochs for training the VAE generator during the first pretrain stage. (Default: 100)")
    
    optional.add_argument('--pretrain1gan',required=False, default='100', help="The number of iterations for training the generator and discriminator jointly during the first pretrain stage. (Default: 100)")
    
    optional.add_argument('--lambdabulkr',required=False, default='1.0', help="Scaling factor for the representative bulk loss. (Default: 1.0)")
    
    optional.add_argument('--pretrain2lr',required=False, default='50', help="The number of epochs for training the VAE generator during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--pretrain2vae',required=False, default='na', help="Sample ID of the target sample when running in single-sample mode.")
    
    optional.add_argument('--pretrain2gan',required=False, default='50', help="The number of iterations for training the generator and discriminator jointly during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--inferepochs',required=False, default='150', help="The number of epochs for training the generator in each mini-stage during the inference. (Default: 150)")
    
    optional.add_argument('--lambdabulkt',required=False, default='8.0', help="Scaling factor for the intial target bulk loss. (Default: 8.0)")
    
    optional.add_argument('--inferlr',required=False, default='2e-4', help="Learning rate during the inference stage. (Default: 2e-4)")
    
    
    
    
    args = parser.parse_args()
    
    
    representatives = args.representatives
    cluster = args.cluster
    targetid = args.targetid
    bulktype = args.bulktype
    lambdad = float(args.lambdad)
    
    pretrain1batch = int(args.pretrain1batch)
    pretrain1lr = float(args.pretrain1lr)
    pretrain1vae = int(args.pretrain1vae)
    pretrain1gan = int(args.pretrain1gan)
    lambdabulkr = float(float9args.lambdabulkr)
    
    pretrain2lr = float(args.pretrain2lr)
    pretrain2vae = int(args.pretrain2vae)
    pretrain2gan = int(args.pretrain2gan)
    inferepochs = int(args.inferepochs)
    lambdabulkt = float(args.lambdabulkt)
    
    inferlr = float(args.inferlr)
    
    scinfer(representatives,cluster,targetid,bulktype,lambdad,pretrain1batch,pretrain1lr,pretrain1vae,pretrain1gan,lambdabulkr,pretrain2lr, pretrain2vae,pretrain2gan,inferepochs,lambdabulkt,inferlr)

if __name__=="__main__":
    main()
