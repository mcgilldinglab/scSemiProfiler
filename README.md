

# scSemiProfiler: Advancing Large-scale Single-cell Studies through Semi-profiling with Deep Generative Models and Active Learning


**scSemiProfiler** is an innovative computational tool that combines deep generative models and active learning to economically generate single-cell data for biological studies. It supports two main application scenarios: **semi-profiling**, which uses deep generative learning and active learning to generate a single-cell cohort with 1/10 to 1/3 sequencing cost, and **single-cell level deconvolution**, which generates single-cell data from bulk data and single-cell references. For more insights, check out our [manuscript in Nature Communications](https://www.nature.com/articles/s41467-024-50150-1), and please consider citing it if you find our method beneficial.

Explore comprehensive details, including API references, usage examples, and tutorials (in [Jupyter notebook](https://jupyter.org/) format), in our [full documentation](https://scsemiprofiler.readthedocs.io/en/latest/) and the README below. 


*Updates:*
- **New Single-Cell Level Deconvolution Pipeline:** A simplified pipeline has been added to scSemiProfiler for generating single-cell data from bulk RNA-seq profiles using a single-cell reference sample. See the [Application Scenarios](#application-scenarios) section for details.

- **Global Mode Functions:** New global mode functions `"inspect_data"` and `"global_stop_checking"` have been introduced. For details, use `print(scSemiProfiler.utils.inspect_data.__doc__)` and `print(scSemiProfiler.utils.global_stop_checking.__doc__)`.

## Table of Contents
- [Application Scenarios](#application-scenarios)
- [Methods Overview](#methods-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Results reproduction](#results-reproduction)
- [Credits and Acknowledgements](#credits-and-acknowledgements)
- [Contacts](#contacts)

## Application Scenarios

### 1. Single-Cell Level Deconvolution
This process allows users to deconvolute bulk RNA-seq data from a target sample into single-cell data, using a single-cell reference sample as a guide. Users need to provide bulk data for both the target and reference samples. The single-cell reference can be derived from real sequencing data or any similar online dataset. Once the pipeline is completed, single-cell data for the target sample is generated and can be used for cell type annotation. This includes de novo annotation or utilizing a classifier trained on the reference data. For further guidance, please refer to the [deconvolution_example.ipynb](deconvolution_example.ipynb).

### 2. Semi-Profile a Cohort
With bulk data for a cohort, select a few representative samples using active learning for real single-cell sequencing and computationally generate single-cell data for the rest target samples. Getting single-cell data using less than 1/3 cost. Example in [example.ipynb](example.ipynb).

---


## Methods Overview
![flowchart](./overview.jpg)

**scSemiProfiler Overview:** scSemiProfiler offers a cost-effective AI-generated alternative to real-profiled single-cell data with high fidelity. 

**a, Curating bulk and reference single-cell data:** Bulk sequencing is performed across the entire cohort. The single-cell reference data can either be provided by the user (e.g., a public reference dataset) or obtained from selected representative samples within the cohort under study. Representative samples can be chosen based on clustering analysis of the bulk data (the global mode of scSemiProfiler) or by using the active learning module. 

**b, In silico inference of target single-cell data from bulk profiles:** For each target sample, a deep generative model first learns the distribution of the reference single-cell data, generating reconstructions of the reference cells. Subsequently, the bulk information of the target sample is incorporated into the cell generation process via fine-tuning, producing single-cell data that matches the target bulk. This AI-powered semi-profiling framework significantly reduces single-cell profiling costs for large cohorts (e.g., a 66.3% savings in the example COVID-19 study). Cost estimates are based on rates from the McGill Genome Centre and costpercell as of 2023. 

**c, High fidelity between cost-effective AI-generated semi-profiled and ground-truth single-cell data:** Left: UMAP visualization shows that the inferred target sample’s single cells (red), generated based on reference cells (blue), closely resemble the real-profiled ground truth of the target sample (red; unknown to the model). Middle: UMAP visualizations compare the real-profiled and semi-profiled COVID-19 cohort with 124 samples, which are similar in terms of cell distribution and cell types (indicated by colors, which are consistent with the legends on the right). Right: Stacked bar plots indicate that the semi-profiled cohort has nearly identical cell type proportions across disease conditions compared to the real-profiled ground truth.

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
	download scSemiProfiler from this repository, go to the downloaded scSemiProfiler package root directory, and use the pip tool to install

	```shell
	pip install .
	```
	
* __Option 2: Install from Github__:    
	```shell
	pip install --upgrade https://github.com/mcgilldinglab/scSemiProfiler/zipball/main
    ```



## Credits and Acknowledgements
**scSemiProfiler** was developed by [Jingtao Wang](https://github.com/JingtaoWang22), [Gregory Fonseca](https://www.mcgill.ca/expmed/dr-gregory-fonseca-0), and [Jun Ding](https://github.com/phoenixding) at McGill University, with support from the Canadian Institutes of Health Research (CIHR), Fonds de recherche du Québec – Santé (FRQS), and the Natural Sciences and Engineering Research Council of Canada (NSERC). Additional funding was provided by the Meakins-Christie Chair in Respiratory Research. This work is part of the Human Cell Atlas (HCA) publication bundle (HCA-8).

## Contacts
Please don't hesitate to contact us if you have any questions and we will be happy to help:
* jingtao.wang at mail.mcgill.ca 
* gregory.fonseca at mcgill.ca
* jun.ding at mcgill.ca



