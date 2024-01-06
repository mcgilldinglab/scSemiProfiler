

# scSemiProfiler: Advancing Large-scale Single-cell Studies through Semi-profiling with Deep Generative Models and Active Learning

scSemiProfiler is an innovative computational tool combining deep generative models and active learning to economically generate single-cell data for biological studies. It efficiently transforms bulk cohort data into detailed single-cell data using templates from selected representative samples. More details are in our [manuscript](https://www.biorxiv.org/content/10.1101/2023.11.20.567929v1). 

For comprehensive details, including API references, usage examples, and tutorials, please refer to our [documentation](https://scsemiprofiler.readthedocs.io/en/latest/).

## Method Overview
![flowchart](./method.jpg)
For an interested cohort, scSemiProfiler runs the following steps to generate single-cell data for all samples.

**a**, Initial Setup: Bulk sequencing is first performed on the entire cohort, with subsequent clustering analysis of this data to pinpoint representative samples, typically those closest to the cluster centroids.

**b**, Representative Profiling: The identified representatives are then subjected to single-cell sequencing. The data obtained from this sequencing is further processed to determine gene set scores and feature importance weights, enriching the subsequent analysis steps.

**c**, Deep Generative Inference: This phase uses a VAE-GAN-based model to estimate single-cell data for a target sample. In its three-stage training, the model initially reconstructs the representative cells, and then produces target cells by analyzing the differences between the two samples as indicated by the bulk data.

**d**, Representative Selection Decision: Decisions are made on selecting additional representatives, considering budget limits and current representative effectiveness. An active learning algorithm, leveraging bulk data and the generative model insights, identifies new optimal representatives. These are then sequenced (**b**) and serve as and integrated as new references in the single-cell inference process (**c**).

**e**, Comprehensive Downstream Analyses: This final panel highlights the extensive analyses possible with semi-profiled single-cell data. It underscores the model’s ability to yield deep, diverse insights, demonstrating the full potential and broad applicability of the semi-profiled data.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Results reproduction](#results-reproduction)
- [Credits](#credits)
- [Contacts](#contacts)


## Prerequisites
First, install [Anaconda](https://www.anaconda.com/). You can find specific instructions for different operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Second, create a new conda environment and activate it:
```
conda create -n semiprofiler python=3.9
```
```
conda activate semiprofiler
```
Finally, install the version of PyTorch compatible with your devices by following the [instructions on the official website](https://pytorch.org/get-started/locally/). 
## Installation

 There are 2 options to install scSemiProfiler.  
* __Option 1: Install from download directory__   
	download scSemiProfiler from this repository, go to the downloaded scSemiProfiler package root directory and use pip tool to install

	```shell
	pip install .
	```
	
* __Option 2: Install from Github__:    
	```shell
	pip install --upgrade https://github.com/mcgilldinglab/scSemiProfiler/zipball/main
    ```


## Results reproduction
Results in our manuscript can be reproduced by running scSemiProfiler on the three datasets we abnalyzed (see below).

### Raw data availability
The preprocessed COVID-19 dataset is from [Stephenson et al.'s study](https://www.nature.com/articles/s41591-021-01329-2) and can be downloaded from Array Express under accession number [E-MTAB-10026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026) The cancer dataset is from [Joanito et al.'s study](https://www.nature.com/articles/s41588-022-01100-4#Sec2). The count expression matrices are available through Synapse under the accession codes [syn26844071](https://www.synapse.org/#!Synapse:syn26844071/wiki/615389) The iMGL dataset is from [Ramaswami1 et al.'s study](https://www.biorxiv.org/content/10.1101/2023.03.09.531934v1.full.pdf).The raw count iMGL bulk and single-cell data can be downloaded from Gene Expression Omnibus (GEO) repository under accesssion number [GSE226081](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226081).

### Preprocessed data availability
The preprocessed single-cell and bulk datasets are available for download [here](https://mcgill-my.sharepoint.com/:u:/g/personal/jingtao_wang_mail_mcgill_ca/ERyixNDuCFNMiL0chjLxij4BtsiRAaFXOnEcCu1lVBDSIQ?e=lCo0YR). These datasets are provided as h5ad files, including normalized counts with cells and genes already filtered, mirroring the format of our example data.

### Experiment details
The three folders contains everything about an older version of scSemiProfiler we used to generate results shown in the manuscript for the three cohorts. The pipeline_test.ipynb in each folder contains code for preprocessing the data and running through most of the functionalities, including representatives' single-cell reconstruction and the single-cell inference for target samples. The semiloop.ipynb in each folder is for semi-profiling the whole cohort using the deep generative model and active learning iteratively. Moreover, semiresultsanalysis.ipynb, deconv_benchmarking.ipynb, and cellchat.ipynb contain code for downstream analyses.

## Credits
scSemiProfiler is jointly developed by [Jingtao Wang](https://github.com/JingtaoWang22), [Gregory Fonseca](https://www.mcgill.ca/expmed/dr-gregory-fonseca-0), and [Jun Ding](https://github.com/phoenixding) from McGill University.

## Contacts
Please don't hesitate to contact us if you have any questions and we will be happy to help:
* jingtao.wang at mail.mcgill.ca 
* gregory.fonseca at mcgill.ca
* jun.ding at mcgill.ca



