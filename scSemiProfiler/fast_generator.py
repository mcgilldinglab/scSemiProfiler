from typing import Callable, Iterable, Optional, List, Union,Dict
import os
from anndata import AnnData
import numpy as np
import collections
from functools import partial
from inspect import getfullargspec, signature

from torch import nn as nn
from torch.nn import ModuleList
import logging
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs)

import scvi
from scvi import REGISTRY_KEYS as _CONSTANTS
from scvi import REGISTRY_KEYS
#from scvi._compat import Literal
from typing import Literal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data, PyroBaseModuleClass 
from scvi.model.base import  UnsupervisedTrainingMixin,ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot
#from scvi.train._metrics import ElboMetric
from scvi.train import  TrainingPlan, TrainRunner#, AdversarialTrainingPlan TrainingPlan,
from scvi.dataloaders import DataSplitter
from scvi.utils import setup_anndata_dsp
from typing import Callable, Iterable, Optional

from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField
)
from scvi.model._utils import _init_library_size
from scvi.module import VAE,Classifier
logger = logging.getLogger(__name__)



import jax
import jax.numpy as jnp
import optax

import pytorch_lightning as pl

from inspect import getfullargspec, signature

#from scvi import ArchesMixin,RNASeqMixin,VAEMixin,UnsupervisedTrainingMixin


scvi._settings.ScviConfig.seed=33


'''
def reparameterize_gaussian(mu, var, bound = 0.0):
    device = mu.device
    std_norm = Normal(torch.zeros(mu.shape),torch.ones(var.shape))
    cdf = torch.zeros(mu.shape)-1
    while (cdf.min() < bound) or (1-cdf.min() < bound):
        z = std_norm.rsample()
        cdf = std_norm.cdf(z)
    
    z = z.to(device)*var.sqrt() + mu
    return z #Normal(mu, var.sqrt()).rsample()
'''

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def identity(x):
    return x

class FCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond


    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)


    def forward(self, x: torch.Tensor, *cat_list: int):

        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x

class myEncoder(nn.Module):
    def __init__(
        self,
        adj,
        geneset_len,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input-geneset_len,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        
        self.g_encoder = FCLayers(
            n_in=geneset_len,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers+1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.geneset_len=geneset_len
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.mean_encoder = nn.Linear(2*n_hidden, n_output)
        self.var_encoder = nn.Linear(2*n_hidden, n_output)
        
        
        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation
   

    
    def forward(self, x: torch.Tensor, neighborx, cellidx,selfw, *cat_list: int,bound=0.0):
        
        device = x.device 
        b = x.shape[0]
        
        
        #neighborx = neighborx.to(device)
        neighborx = neighborx.reshape((b,-1))
        
        # Parameters for latent distribution
        '''print()
        print('self.n_input',self.n_input)
        print('neighborx')
        print(neighborx.device)
        print(neighborx.shape)
        print(neighborx.dtype)
        #print('catlist')
        #print(*cat_list)
        #print(cat_list)
        print('x')
        print(x.shape)
        print(x.dtype)
        print(x.device)
        print(x)
        print(1)
        print(neighborx)
        print(x.shape)'''


        g = x[:,-self.geneset_len:]
        g = self.g_encoder(g)
        
        q = self.encoder(neighborx, *cat_list)
        
        q = torch.cat([q,g],1)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent



# Decoder
class myDecoderSCVI(nn.Module):

    def __init__(
        self,
        geneset_len : int,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output-geneset_len),
            nn.Softmax(dim=-1),
        )
        
        self.geneset_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, geneset_len),
            nn.Softmax(dim=-1),
        )
        
        # for graph
        
        self.w = torch.nn.Parameter(torch.randn((n_output,n_output))/100)
        self.proj=nn.Linear(n_output,n_output)
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
    
        self.geneset_len=geneset_len

    
    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, glibrary: torch.Tensor, *cat_list: int ):

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        
        px_scale = self.px_scale_decoder(px)
        g_scale = self.geneset_scale_decoder(px)
        
        px_dropout = self.px_dropout_decoder(px)
        #self.updatew()
        
        #px_dropout =  px_dropout + torch.matmul(px_dropout,adj*w2)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        #g = px_rate[:,-self.geneset_len:]
        g = torch.exp(glibrary) * g_scale
        px_rate = torch.cat([px_rate,g],axis=1)
        
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout, g








torch.backends.cudnn.benchmark = True

class myVAE(BaseModuleClass):

    def __init__(
        self,
        adj,
        variances,
        markermask,
        bulk,
        geneset_len,
        adata,
        n_input: int,
        countbulkweight: float = 1,
        power:float=2,
        upperbound:float=99999,
        logbulkweight: float = 0,
        absbulkweight: float = 0,
        abslogbulkweight:float=0,
        corrbulkweight:float = 0,
        meanbias:float = 0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.variances = variances
        self.markermask = markermask
        self.bulk = bulk
        self.geneset_len = geneset_len
        self._adata=adata
        self.n_input = n_input 
        
        self.logbulkweight = logbulkweight
        self.absbulkweight=absbulkweight
        self.abslogbulkweight=abslogbulkweight
        self.corrbulkweight=corrbulkweight
        self.meanbias=meanbias
        self.countbulkweight = countbulkweight
        self.power=power
        self.upperbound=upperbound
        self.use_observed_lib_size = use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )


        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input-geneset_len, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input-geneset_len, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = myEncoder(
            adj,
            geneset_len,
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = myEncoder(
            adj,
            geneset_len,
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = myDecoderSCVI(
            geneset_len,
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        

        cellidx = tensors['cellidx']  # (batch)
        cellidx = cellidx.type(torch.LongTensor)
        #cellidx = cellidx.to(x.device)
        
        neighborx =  tensors['neighborx'] 
        #neighborx = neighborx[cellidx]
        
        
        input_dict = dict(
            x=x,  neighborx=neighborx, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        glibrary = inference_outputs["glibrary"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        input_dict = {
            "z": z,
            "library": library,
            "glibrary":glibrary,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    '''
    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars'''

    @auto_move_data
    def inference(self, x, neighborx, batch_index, cont_covs=None, cat_covs=None, n_samples=1,bound=0.0):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        
        x_ = x
        genelen = neighborx.shape[1]
        totallen = x.shape[1]
        if self.use_observed_lib_size:
            library = torch.log(x_[:,:genelen].sum(1)).unsqueeze(1)
            glibrary = torch.log(x_[:,genelen:].sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            neighborx = torch.log(1 + neighborx)
            awe=1+1

        encoder_input = x_
        qz_m, qz_v, z =  self.z_encoder(encoder_input, neighborx, batch_index,bound) #, *categorical_input,bound)
                                 
        ql_m, ql_v = None, None

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
            glibrary = glibrary.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )


        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library,glibrary=glibrary)
        return outputs


    @auto_move_data
    def generative(
        self,
        z,
        library,
        glibrary,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_scale, px_r, px_rate, px_dropout,g = self.decoder(
            self.dispersion, decoder_input, library, glibrary, batch_index, *categorical_input, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )


    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)

        kl_divergence_l = 0.0
        variances = self.variances
        markermask = self.markermask
        bulk = self.bulk
        reconst_loss, bulk_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout,\
                                                    variances,markermask,bulk)
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        
        
        #print(bulk)
        #print(type(bulk))
        if (type(bulk) == type(None)):
            #print(0)
            return LossOutput(loss, reconst_loss, kl_local,kl_global)
        else:
            return LossOutput(loss, reconst_loss, kl_local,bulk_loss)# kl_global)

    @torch.no_grad()
    def nb_sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
        bound=20.0,
    ) -> np.ndarray:
        inference_kwargs = dict(n_samples=n_samples,bound=bound)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        dist = NegativeBinomial(mu=px_rate, theta=px_r)
        mdist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        mask = (mdist.mean>bound)
        exprs = dist.mean  # * mask
        
        return exprs.cpu(), mask
        

    @torch.no_grad()
    def stoch_sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
        bound=0.0,
    ) -> np.ndarray:

        inference_kwargs = dict(n_samples=n_samples,bound=bound)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()
        
    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
        bound=0.0,
    ) -> np.ndarray:
 
        inference_kwargs = dict(n_samples=n_samples,bound=bound)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.mean #dist.sample()

        return exprs.cpu()


    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, variances, markermask, bulk) -> torch.Tensor:
        if type(variances) != type(None):
            normv = variances
            ##########markermask = markermask
            vs,idx=normv.sort()
            threshold =  vs[5128//2] #normv.mean()
            #threshold2 =  vs[3540]
            #normv = torch.tensor((normv>normv.mean())) 
            normv = torch.tensor((normv>threshold)) 
            #msk = torch.tensor((normv>threshold2)) 
            normv = 0.5*normv + 1
            #normv = torch.ones
            ####normv = normv + markermask*0.5
            normv = torch.tensor(normv)
        
        if self.gene_likelihood == "zinb":
            reconst_loss =    ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                ).log_prob(x)
            #reconst_loss *= normv
            reconst_loss = -reconst_loss.sum(dim=-1)
            
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        
        
        if type(bulk)!= type(None):
            predicted_batch_mean = ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                ).mean
            
            
            
            # norm total batch mean
            '''
            predicted_batch_mean = (predicted_batch_mean*1e4).permute((1,0)) / predicted_batch_mean.sum(axis=-1)
            
            predicted_batch_mean = (predicted_batch_mean-self.meanbias)
            predicted_batch_mean = predicted_batch_mean*(predicted_batch_mean>0)
            
            predicted_batch_mean = predicted_batch_mean.permute((1,0))
            '''
    
            predicted_batch_mean = predicted_batch_mean.mean(axis=0) # average over batch dimension
                                                                     # shape should be (gene) now 
            
            #print(predicted_batch_mean.shape, 111)
            
            predicted_batch_mean = predicted_batch_mean.reshape((-1))[:-self.geneset_len]
            
            #print(predicted_batch_mean.shape, 222)
            
            bulk = bulk.reshape((-1))[:-self.geneset_len]
            
            #print(bulk.shape, 333)
            
            bulk = bulk.to(predicted_batch_mean.device)
            predicted_batch_mean = predicted_batch_mean[:len(bulk)]
            
            #print(predicted_batch_mean.shape, 444)
            
            #print(predicted_batch_mean.shape)
            
            #print(bulk.shape)
           # cp = torch.nn.functional.normalize(predicted_batch_mean,dim=0)
           # cb = torch.nn.functional.normalize(bulk,dim=0)
            
            
            ### expression transformation for bulk loss
            
            bulk_loss = self.countbulkweight * (predicted_batch_mean - bulk)**self.power + \
                        self.logbulkweight * torch.abs(torch.log(predicted_batch_mean+1) - torch.log(bulk+1)) +  \
                        self.absbulkweight * torch.abs(predicted_batch_mean - bulk) + \
                        self.abslogbulkweight * torch.abs(torch.log(predicted_batch_mean+1) - torch.log(bulk+1)) #+ \
                       # self.corrbulkweight * -(cp*cb).sum()/  ((cb**2).sum())**0.5 *  (((cp**2).sum())**0.5).sum()
            
            #bulk_loss = bulk_loss * (predicted_batch_mean > self.meanbias)
            bulk_loss = bulk_loss * (predicted_batch_mean < self.upperbound)
            
            bulk_loss = bulk_loss.mean() # average over genes

            reconst_loss = reconst_loss + bulk_loss

        else:
            bulk_loss = 0
        
        return reconst_loss, bulk_loss


    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]
            glibrary = inference_output["glibrary"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            log_prob_sum += p_z + p_x_zl

            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            log_prob_sum -= q_z_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl



logger = logging.getLogger(__name__)

class D(torch.nn.Module):
    
    def __init__(self,indim,hdim=128):    
        super(D,self).__init__()
        self.l1 = torch.nn.Linear(indim,2*hdim)
        self.act1 = torch.nn.LeakyReLU()    #torch.nn.GELU()
        
        self.l11 = torch.nn.Linear(2*hdim,hdim)
        self.act11 = torch.nn.LeakyReLU()
        
        self.l2 = torch.nn.Linear(hdim,10)
        self.act2 = torch.nn.LeakyReLU()    #torch.nn.GELU()
        self.l3 = torch.nn.Linear(10,1)
        self.sig = torch.nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
        
    def forward(self,x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l11(x)
        x = self.act11(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        x = self.sig(x)
        return x


class AdversarialTrainingPlan(TrainingPlan):

    def __init__(
        self,
        module: BaseModuleClass,
        lr=1e-3,
        lr2=1e-3,
        kappa = 4040*0.001,
        weight_decay=1e-6,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = False,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        clip = None,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        self.n_output_classifier = 1
        self.lr2=lr2
        self.kappa = kappa
        self.adversarial_classifier = adversarial_classifier
        
        self.scale_adversarial_loss = scale_adversarial_loss
        self.clip = clip
        
    def loss_adversarial_classifier(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
        
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        # update discriminator every step
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
            #for p in self.adversarial_classifier.parameters():
            #    p.data.clamp_(-1, 1)
            
        # update generator every 5 steps
        if optimizer_idx == 0:
            if True: #(batch_idx + 1) % 2 == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                

                    
                optimizer.step(closure=optimizer_closure)
            else:
                #if type(self.clip)!=type(None):
                #    print(self.clip)
                #    torch.nn.utils.clip_grad_norm(self.module.parameters(), self.clip)
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
    
    
    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        if optimizer_idx == 0:
            # Lightning will handle the gradient clipping
            
            if type(self.clip)!=type(None):
                #print(self.clip)
                #torch.nn.utils.clip_grad_norm(self.module.parameters(), self.clip)
                self.clip_gradients(
                    optimizer, gradient_clip_val=self.clip, gradient_clip_algorithm=gradient_clip_algorithm
                )
        #elif optimizer_idx == 1:
        #    self.clip_gradients(
        #        optimizer, gradient_clip_val=gradient_clip_val * 2, gradient_clip_algorithm=gradient_clip_algorithm
        #    )     


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = self.kappa #4040 * 0.1
        
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
 
        if optimizer_idx == 0:
          #  self.module.train(True)
            #self.adversarial_classifier.train(False)
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch, loss_kwargs=self.loss_kwargs
            )
            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:
                
                px_scale = generative_outputs['px_scale']
                px_r = generative_outputs['px_r']
                px_rate = generative_outputs['px_rate']
                px_dropout = generative_outputs['px_dropout']
                
                
                dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                xh = dist.mean

                
                valid = torch.ones(xh.size(0), 1)
                valid = valid.type_as(xh)
                fool_loss = self.loss_adversarial_classifier(self.adversarial_classifier(xh), valid) * kappa
                loss = loss + fool_loss 

            self.log("train_loss", loss, on_epoch=True)
            self.log("fool_loss", fool_loss, on_epoch=True)
            
            #self.compute_and_log_metrics(scvi_loss, self.elbo_train,mode='train')
            self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
            return loss
        
        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:

            # xh = self.module.sample(batch)
            
            # negative binomial version sampling
            xh,__ = self.module.nb_sample(batch)
            xh = xh.to(self.module.device)
            x = batch[_CONSTANTS.X_KEY]
            
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(x)
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(x)

            
            fake_loss = self.loss_adversarial_classifier(self.adversarial_classifier(xh), fake)
            true_loss = self.loss_adversarial_classifier(self.adversarial_classifier(x), valid)
                                                             
            loss = (fake_loss+true_loss)/2
            
            self.log("gan_loss", loss, on_epoch=True)
            self.log("gan_true_loss", true_loss, on_epoch=True)
            self.log("gan_fake_loss", fake_loss, on_epoch=True)
            
            return loss

        
    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
  
        optimizer1 = torch.optim.Adam(
            params1, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(
                lambda p: p.requires_grad, self.adversarial_classifier.parameters()
            )
            optimizer2 = torch.optim.Adam(
                params2, lr=self.lr2, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # bug in pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                config1["scheduler"] = config1.pop("lr_scheduler")
                scheds = [config1]
                return opts, scheds
            else:
                return opts

        return config1

    
    
class fastgenerator(
    BaseModelClass,VAEMixin
):


    def __init__(
        self,
        adj,
        variances,
        markermask,
        bulk,
        geneset_len,
        adata: AnnData,
        clip = None,
        countbulkweight: float = 1,
        power:float = 2.0,
        upperbound:float = 99999,
        logbulkweight: float = 0,
        absbulkweight:float=0,
        abslogbulkweight:float=0,
        corrbulkweight:float=0,
        meanbias:float=0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(fastgenerator, self).__init__(adata)

        n_cats_per_cov = None#(
        #    self.scvi_setup_dict_["extra_categoricals"]["n_cats_per_key"]
        #    if "extra_categoricals" in self.scvi_setup_dict_
        #    else None
        #)
        self.module = myVAE(
            adj,
            variances,
            markermask,
            bulk,
            geneset_len,
            self._adata,
            countbulkweight = countbulkweight,
            power=power,
            upperbound=upperbound,
            logbulkweight = logbulkweight,
            absbulkweight=absbulkweight,
            abslogbulkweight=abslogbulkweight,
            corrbulkweight=corrbulkweight,
            meanbias=meanbias,
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            #n_continuous_cov=self.summary_stats["n_continuous_covs"],
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self.adversarial_classifier = D(indim = self.module.n_input)
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())
        self.clip = clip
        
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
    
        
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = AdversarialTrainingPlan(self.module, 
                                                adversarial_classifier=self.adversarial_classifier  ,
                                                clip=self.clip
                                                , **plan_kwargs)
        self.training_plan = training_plan
        
        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
    
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            ObsmField(
                'neighborx', 'neighborx'
            ),
            NumericalObsField(
                'cellidx', 'cellidx'
            ),
            NumericalObsField(
                'selfw', 'selfw'
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
print('scVI ready')



