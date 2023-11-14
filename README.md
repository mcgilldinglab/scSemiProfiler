

# scSemiProfiler: Advancing Large-scale Single-cell Studies through Semi-profiling with Deep Generative Models and Active Learning
## Introduction
scSemiProfiler is an innovative computational framework that marries deep generative model with active learning strategies to provide affordable single-cell data for large-scale single-cell studies. This method adeptly infers single-cell profiles across large cohorts by fusing bulk sequencing data with targeted single-cell sequencing from a carefully chosen representatives. Developed initially for large disease cohorts,scSemiProfiler is adaptable for broad applications, offering a scalable, cost-effective solution for single-cell profiling.

![flowchart](./method.jpg)
For an interested cohort, scSemiProfiler runs the following pipeline to generate single-cell data.

**a**, Initial Configuration: Bulk sequencing is initially conducted on the entire cohort, followed by a clustering analysis of this bulk data. This analysis serves to identify representative samples, usually those nearest to the cluster centroids.

**b**, Representative Profiling: The identified representatives are then subjected to single-cell sequencing. The data obtained from this sequencing is further processed to determine gene set scores and feature importance weights, enriching the subsequent analysis steps.

**c**, Deep Generative Inference: Utilizing a VAE-GAN based model, the process integrates comprehensive bulk data from the cohort with the single-cell data derived from the representatives. During the model's 3-stage training, the generator aims to optimize losses $L_{G_{Pretrain1}}$, $L_{G_{Pretrain2}}$, and $L_{inference}$, respective, whereas the discriminator focuses on minimizing $L_{D}$. In $L_{D}$, $G$ and $D$ are the generator and discriminator respectively. $D((\mathbf{x_{i}},\mathbf{s_{i}}))$ is the generator's predicted probability of the input cell being real when it is indeed real.  $D(G((\mathbf{x_{i}},\mathbf{s_{i}})))$ is the generator's predicted probability of the input cell being real when it is a generator's reconstructed cell. 

**d**, Representative Selection Decision: Decisions on further representative selection are made, taking into account budget constraints and the effectiveness of the current representatives. An active learning algorithm, which draws on insights from the bulk data and the generative model, is employed to pinpoint additional optimal representatives. These newly selected representatives then undergo further single-cell sequencing (**b**) and serve as new reference points for the ongoing semi-profiling process **c**).

**e**, Comprehensive Downstream Analyses: The final stage involves conducting extensive downstream analyses using the semi-profiled single-cell data. This phase is pivotal in demonstrating the model’s capacity to provide deep and wide-ranging insights, showcasing the full potential and applicability of the semi-profiled data.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Results reproduction](#Results-reproduction)
- [Credits](#Credits)

## Used software information 
* Python == 3.9.13
* Python side-packages:   
-- pytorch == 1.12.1  
-- numpy == 1.19.2     
-- scanpy == 1.9.8  
-- scikit-learn == 1.1.2  
-- faiss >= 1.7.2  
-- scvi == 0.17.4  
-- pytorch_lightning == 1.7.7

## Results reproduction
The three folders correspond to the everything relevant to the three cohorts we used to examine the performance of scSemiProfiler. 

### Raw data availability
The preprocessed COVID-19 dataset is from [Stephenson et al.'s study](https://www.nature.com/articles/s41591-021-01329-2) and can be downloaded from Array Express under accession number \href{https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026}{E-MTAB-10026} The cancer dataset is from [Joanito et al.'s study](https://www.nature.com/articles/s41588-022-01100-4#Sec2). The count expression matrices are available through Synapse under the accession codes \href{https://www.synapse.org/#!Synapse:syn26844071/wiki/615389}{syn26844071} The iMGL dataset is from [Ramaswami1 et al.'s study](https://www.biorxiv.org/content/10.1101/2023.03.09.531934v1.full.pdf).The raw count iMGL bulk and single-cell data can be downloaded from Gene Expression Omnibus (GEO) repository under accesssion number \href{https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226081}{GSE226081}.

 
### Testing functionalities on a few example samples
The pipeline_test,ipynb in each folder contains code for preprocessing the data and running through most of the functionalities, including representatives' single-cell reconstruction and the single-cell inference for target samples.
### Perform semi-profiling for a cohort
The semiloop.ipynb is for semi-profiling the whole cohort using the deep generative model and active learning iteratively. 
### Downstream analysis results generation
In each folder, semiresultsanalysis.ipynb, deconv_benchmarking.ipynb, and cellchat.ipynb contain code for most downstream analysis.

## Credits
scSemiProfiler is jointly developed by [Jingtao Wang](https://github.com/JingtaoWang22), [Gregory Fonseca](https://www.mcgill.ca/expmed/dr-gregory-fonseca-0), and [Jun Ding](https://github.com/phoenixding) from McGill University.

## Contacts
* jingtao.wang at mail.mcgill.ca 



