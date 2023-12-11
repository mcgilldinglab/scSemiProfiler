import numpy as np
import torch
import os
import timeit
import copy
import argparse
import anndata
from anndata import AnnData
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
import gc
from fast_generator import *
import pickle

def setdata(name,sid,device='cuda:0',k=15,diagw=1.0):
    adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad') 
    
    # load geneset
    sample_geneset = np.load(name + '/geneset_scores/'+sid+'.npy')
    setmask = np.load(name + '/hvset.npy')
    sample_geneset = sample_geneset[:,setmask]
    sample_geneset = sample_geneset.astype('float32')
    geneset_len = sample_geneset.shape[1]
    
    features = np.concatenate([adata.X,sample_geneset],1)
    bdata = anndata.AnnData(features,dtype='float32')
    bdata.obs = adata.obs
    bdata.obsm = adata.obsm
    bdata.uns = adata.uns
    adata = bdata.copy()
    
    # adj for cell graph
    adj = adata.obsm['adj']
    adj = torch.from_numpy(adj.astype('float32'))
    
    # variances
    variances = torch.tensor(adata.uns['feature_var'])
    variances = variances.to(device)
    
    #pseudobulk
    pseudobulk = np.array(adata.X.mean(axis=0)).reshape((-1))
    fastgenerator.setup_anndata(adata)

    return adata,adj,variances,pseudobulk,geneset_len




def fastrecon(name, sid, device='cuda:0',k=15,diagw=1.0,vaesteps=100,gansteps=100,lr=1e-3,save=True,path=None):
    
    #set data
    adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
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
            if (os.path.isdir(name + '/models')) == False:
                os.system('mkdir '+ name + '/models')
            path = name + '/models/fast_reconst1_'+sid
        torch.save(model.module.state_dict(), path)
    
    
    with open(name+'/history/pretrain1_' + sid + '.pkl', 'wb') as pickle_file:
                        pickle.dump(model.history, pickle_file)
            
    return model


# reconst stage 2
def reconst_pretrain2(name, sid ,premodel,device='cuda:0',k=15,diagw=1.0,vaesteps=50,gansteps=50,lr=1e-4,save=True,path=None):
    adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
    
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
        pass
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
            path = name + '/models/fastreconst2_' + sid
        torch.save(model.module.state_dict(), path)
    
    with open(name+'/history/pretrain2_' + sid + '.pkl', 'wb') as pickle_file:
                        pickle.dump(model.history, pickle_file)
    
    return model





def unisemi0(name,adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model0 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight =1*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                      power=2,upperbound=99999,meanbias=0)
    if type(premodel)==type('string'):
        model0.module.load_state_dict(torch.load(premodel))
    else:
        model0.module.load_state_dict(premodel.module.state_dict())
    lr = 2e-4 
    model0.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model0.module.state_dict(), name+'/tmp/model0')
    return model0.history


def unisemi1(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model1 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 4*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model1.module.load_state_dict(torch.load(name+'/tmp/model0'))
    lr = 2e-4
    model1.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model1.module.state_dict(), name+'/tmp/model1')
    return model1.history

def unisemi2(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model2 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 16*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model2.module.load_state_dict(torch.load(name+'/tmp/model1'))
    lr = 2e-4
    model2.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model2.module.state_dict(), name+'/tmp/model2')
    return model2.history

def unisemi3(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model3 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 64*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model3.module.load_state_dict(torch.load(name+'/tmp/model2'))
    lr = 2e-4 
    model3.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model3.module.state_dict(), name+'/tmp/model3')
    return model3.history

def unisemi4(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model4 = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 128*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model4.module.load_state_dict(torch.load(name+'/tmp/model3'))
    lr = 2e-4 
    model4.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model4.module.state_dict(), name+'/tmp/model4')
    return model4.history

def unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0):
    model = fastgenerator(adata=adata,adj=adj,variances=variances,geneset_len=geneset_len,\
                      markermask=None,bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound,meanbias=0)
    model.module.load_state_dict(torch.load(name+'/tmp/model4'))
    lr = 2e-4 
    model.train(max_epochs=150, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model.module.state_dict(), name+'/tmp/model')
    return model.history

def fast_semi(name,reprepid,tgtpid,premodel,device='cuda:0',k=15,diagw=1.0):
    sids = []
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    adata,adj,variances,reprepseudobulk,geneset_len = setdata(name,sids[reprepid],device=device,k=k,diagw=diagw)
    
    varainces = None
    
    maxexpr = adata.X.max()
    upperbounds = [maxexpr/2, maxexpr/4, maxexpr/8, maxexpr/(8*np.sqrt(2)),maxexpr/16, maxexpr/32,maxexpr/64]     
    
    genelen = len(np.load(name+'/hvgenes.npy',allow_pickle=True))
    
    #(5) tgt bulk
    bulkdata = anndata.read_h5ad(name + '/processed_bulkdata.h5ad')
    tgtbulk = np.exp(bulkdata.X[tgtpid]) - 1
    tgtbulk = np.array(tgtbulk).reshape((1,-1))
    bulk = adata.X.mean(axis=0)
    bulk = np.array(bulk).reshape((1,-1))
    bulk[:,:tgtbulk.shape[1]] = tgtbulk
    bulk = torch.tensor(bulk).to(device)
    
    batch_size=int(np.min([adata.X.shape[0],9000]))
    
    
    #(6) semiprofiling
    fastgenerator.setup_anndata(adata)

    
    hist = unisemi0(name,adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)
    histdic={}
    histdic['total0'] = hist['train_loss_epoch']
    histdic['bulk0'] = hist['kl_global_train']
    #del premodel
    gc.collect()
    torch.cuda.empty_cache() 
    #import time
    #time.sleep(10)
    hist = unisemi1(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[0],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total1'] = hist['train_loss_epoch']
    histdic['bulk1'] = hist['kl_global_train']
    #del model0
    gc.collect()
    torch.cuda.empty_cache() 
    hist = unisemi2(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[1],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total2'] = hist['train_loss_epoch']
    histdic['bulk2'] = hist['kl_global_train']
    #del model1
    gc.collect()

    #time.sleep(10)
    torch.cuda.empty_cache() 
    
    hist = unisemi3(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[2],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

    histdic['total3'] = hist['train_loss_epoch']
    histdic['bulk3'] = hist['kl_global_train']
    
    
    #del model2
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    hist = unisemi4(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[3],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0)

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
    model.module.load_state_dict(torch.load(name+'/tmp/model4'))

    # inference
    xsemi = []
    scdl = model._make_data_loader(
            adata=adata,batch_size=batch_size
    )
    for tensors in scdl:
        samples = model.module.sample(tensors, n_samples=1)
        xsemi.append(samples)
    
    # save inferred data
    xsemi = np.array(torch.cat(xsemi))[:,:genelen]
    torch.save(model.module.state_dict(), name+'/models/semi_'+sids[reprepid]+"_to_"+sids[tgtpid])
    xsemi = xsemi*(xsemi>10)
    np.save(name + '/inferreddata/'+ sids[reprepid]+'_to_'+sids[tgtpid],xsemi)
    
    # save training history
    with open(name+'/history/inference_' + sids[reprepid] + '_to_' + sids[tgtpid] + '.pkl', 'wb') as pickle_file:
                    pickle.dump(histdic, pickle_file)
    
    
    gc.collect()
    torch.cuda.empty_cache() 
    
    
    return histdic,xsemi,model


def scinfer(name, representatives,cluster,targetid,bulktype,
    lambdad = 4.0,
    pretrain1batch = 128,
    pretrain1lr = 1e-3,
    pretrain1vae = 100,
    pretrain1gan = 100,
    lambdabulkr = 1,
    pretrain2lr = 1e-4,
    pretrain2vae = 50,
    pretrain2gan = 50,
    inferepochs = 150,
    lambdabulkt = 8.0,
    inferlr = 2e-4,
    device = 'cuda:0'):
    
    if (os.path.isdir(name + '/inferreddata')) == False:
        os.system('mkdir ' + name + '/inferreddata')
    if (os.path.isdir(name + '/models')) == False:
        os.system('mkdir ' + name + '/models')
    if (os.path.isdir(name + '/tmp')) == False:
        os.system('mkdir ' + name + '/tmp')
    if (os.path.isdir(name + '/history')) == False:
        os.system('mkdir '+ name + '/history')
    
    device = device
    
    
    k = 15
    diagw = 1.0
    if (representatives[-3:] == 'txt') or type(representatives)==type([]):
        print('Start single-cell inference in cohort mode')
        
        
        sids = []
        f = open(name + '/sids.txt','r')
        lines = f.readlines()
        for l in lines:
            sids.append(l.strip())
        f.close()


        repres = []
        f=open(representatives,'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            repres.append(int(l.strip()))

        cluster_labels = []
        f=open(cluster,'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            cluster_labels.append(int(l.strip()))
        
        #timing
        pretrain1start = timeit.default_timer()
        
        print('pretrain 1: representative reconstruction')
        repremodels = []
        for rp in repres:
            sid = sids[rp]
            
            # if exists, load model
            modelfile = 'fast_reconst1_' + sid
            path = name + '/models/fast_reconst1_'+sid
            if modelfile in os.listdir(name + '/models'):
                print('load existing pretrain 1 reconstruction model for '+sid)
                adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
                model = fastgenerator(adj,variances,None,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
                model.module.load_state_dict(torch.load(path))
                repremodels.append(model)
                #continue
            else:
                # otherwise, train model
                repremodels.append(\
                                   fastrecon(name=name,sid=sid,device=device,k=15,diagw=1,vaesteps=int(pretrain1vae),gansteps=int(pretrain1gan),save=True,path=None)\
                                  )

        # timing
        pretrain1end = timeit.default_timer()
        f=open('pretrain1time.txt','w')
        f.write(str(pretrain1end-pretrain1start))
        f.close()
        
        #timing
        pretrain2start = timeit.default_timer()
        
        print('pretrain2: reconstruction with representative bulk loss')
        repremodels2=[]
        i=0
        for rp in repres:
            sid = sids[rp]
            # if exists, load model
            print('load existing model')
            modelfile = 'fastreconst2_' + sid
            path = name + '/models/fastreconst2_' + sid 
            if modelfile in os.listdir(name + '/models'):
                print('load existing pretrain 2 model for ' + sid)
                adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
                model = fastgenerator(adj,variances,None,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
                model.module.load_state_dict(torch.load(path))
                repremodels2.append(model)
                #continue
            else:
                repremodels2.append(reconst_pretrain2(name,sid,repremodels[i],device,k=15,diagw=1.0,vaesteps=int(pretrain2vae),gansteps=int(pretrain2gan),save=True))
            i=i+1
            
        #timing
        pretrain2end = timeit.default_timer()
        f=open('pretrain2time.txt','w')
        f.write(str(pretrain2end-pretrain2start))
        f.close()
        
        
        #timing
        f = open('infertime.txt','w')
        
        print('inference')
        for i in range(len(sids)):
            if i not in repres:
                #timing
                inferstart = timeit.default_timer()
                
                tgtpid = i
                reprepid = repres[cluster_labels[i]]
                
                fname = sids[reprepid]+'_to_'+sids[tgtpid]+'.npy'
                if fname in os.listdir(name+'/inferreddata/'):
                    print('Inference for '+sids[i]+' has been finished previously. Skip.')
                    continue
                
                premodel = repremodels2[cluster_labels[i]]
                histdic,xsemi,infer_model  = fast_semi(name,reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
                
                
                #timing
                inferend = timeit.default_timer()
                f.write(str(inferend-inferend)+'\n')
        #timing
        f.close()
        
    else:
        print('Start single-cell inference in single-sample mode')
        
        repre = representatives
        target = targetid
        print('pretrain1: reconstruction')
        repremodel = fastrecon(pid=representatives, tgtpid=None,device=device,k=15,diagw=1,vaesteps=int(pretrain1vae),gansteps=int(pretrain1gan),save=True,path=None)
        print('pretrain2: reconstruction with representative bulk loss')
        premodel = reconst_pretrain2(repre,repremodel,device,k=15,diagw=1.0,vaesteps=int(pretrain2vae),gansteps=int(pretrain2gan),save=True)
        print('inference')
        fast_semi(repre,target,premodel,device=device,k=15,diagw=1.0)
    
    print('Finished single-cell inference')
    return




def main():
    parser = argparse.ArgumentParser(description="scSemiProfiler scinfer")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--name',required=True,help="Project name (same as previous steps).")
    
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
    
    optional.add_argument('--pretrain2lr',required=False, default='1e-4', help="The number of epochs for training the VAE generator during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--pretrain2vae',required=False, default='50', help="Sample ID of the target sample when running in single-sample mode.")
    
    optional.add_argument('--pretrain2gan',required=False, default='50', help="The number of iterations for training the generator and discriminator jointly during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--inferepochs',required=False, default='150', help="The number of epochs for training the generator in each mini-stage during the inference. (Default: 150)")
    
    optional.add_argument('--lambdabulkt',required=False, default='8.0', help="Scaling factor for the intial target bulk loss. (Default: 8.0)")
    
    optional.add_argument('--inferlr',required=False, default='2e-4', help="Learning rate during the inference stage. (Default: 2e-4)")
    
    
    
    
    args = parser.parse_args()
    
    name = args.name
    representatives = args.representatives
    cluster = args.cluster
    targetid = args.targetid
    bulktype = args.bulktype
    lambdad = float(args.lambdad)
    
    pretrain1batch = int(args.pretrain1batch)
    pretrain1lr = float(args.pretrain1lr)
    pretrain1vae = int(args.pretrain1vae)
    pretrain1gan = int(args.pretrain1gan)
    lambdabulkr = float(args.lambdabulkr)
    
    pretrain2lr = float(args.pretrain2lr)
    pretrain2vae = int(args.pretrain2vae)
    pretrain2gan = int(args.pretrain2gan)
    inferepochs = int(args.inferepochs)
    lambdabulkt = float(args.lambdabulkt)
    
    inferlr = float(args.inferlr)
    
    
    
    scinfer(name, representatives,cluster,targetid,bulktype,lambdad,pretrain1batch,pretrain1lr,pretrain1vae,pretrain1gan,lambdabulkr,pretrain2lr, pretrain2vae,pretrain2gan,inferepochs,lambdabulkt,inferlr)

if __name__=="__main__":
    main()
