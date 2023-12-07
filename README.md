

# scSemiProfiler: Advancing Large-scale Single-cell Studies through Semi-profiling with Deep Generative Models and Active Learning

scSemiProfiler is an innovative computational tool combining deep generative models and active learning to economically generate single-cell data for biological studies. It efficiently transforms bulk cohort data into detailed single-cell data using templates from selected representative samples. More details are in our [paper](https://www.biorxiv.org/content/10.1101/2023.11.20.567929v1). 

## Method Overview
![flowchart](./method.jpg)
For an interested cohort, scSemiProfiler runs the following steps to generate single-cell data for all samples.

**a**, Initial Setup: Bulk sequencing is first performed on the entire cohort, with subsequent clustering analysis of this data to pinpoint representative samples, typically those closest to the cluster centroids.

**b**, Representative Profiling: The identified representatives are then subjected to single-cell sequencing. The data obtained from this sequencing is further processed to determine gene set scores and feature importance weights, enriching the subsequent analysis steps.

**c**, Deep Generative Inference: This phase uses a VAE-GAN-based model to estimate single-cell data for a target sample. In its three-stage training, the model initially reconstructs the representative cells, and then produces target cells by analyzing the differences between the two samples as indicated by the bulk data.

**d**, Representative Selection Decision: Decisions are made on selecting additional representatives, considering budget limits and current representative effectiveness. An active learning algorithm, leveraging bulk data and the generative model insights, identifies new optimal representatives. These are then sequenced (**b**) and serve as and integrated as new references in the single-cell inference process (**c**).

**e**, Comprehensive Downstream Analyses: This final panel highlights the extensive analyses possible with semi-profiled single-cell data. It underscores the model’s ability to yield deep, diverse insights, demonstrating the full potential and broad applicability of the semi-profiled data.


## Table of Contents
- [Software](#software)
- [Installation](#installation)
- [Usage](#Usage)
- [Example](#Example)
- [Results reproduction](#Results-reproduction)
- [Credits](#Credits)

## Software
* Python3.9+
* Python side-packages:   
-- scanpy >= 1.9.6  
-- scipy >= 1.11.4  
-- anndata >= 0.10.3  
-- faiss-cpu >= 1.7.4  
-- torch >= 1.12.1  
-- scikit-learn >= 1.3.2  
-- pandas >= 2.1.3  
-- jax >= 0.4.19  
-- scvi-tools >= 1.0.4  

## Installation
 Highly recommended:    
 Install conda and create a new conda environment:
```
conda create -n semiprofiler python=3.9
```
Then activate the conda environment
```
conda activate semiprofiler
```

 There are 2 options to install scSemiProfiler.  
* __Option 1: Install from download directory__   
	cd to the downloaded scSemiProfiler package root directory

	```shell
	cd scSemiProfiler
	```

    use pip tool to install

	```shell
	pip install .
	```

    or

	run python setup to install   

	```shell
	python setup.py install
	```
	
* __Option 2: Install from Github__:    

	python 3: 
	```shell
	pip install --upgrade https://github.com/mcgilldinglab/scSemiProfiler/zipball/main
## Usage
In this section, we provide guidance on executing each step of scSemiProfiler with your dataset. scSemiProfiler offers two modes of operation: it can be run via the command line or imported as a Python package. For every command line instruction provided below, there is an equivalent Python function. Detailed usage examples of these functions can be found in the "example.ipynb" notebook.

**a,** Initial Setup

For this initial configuration step, simply provide your bulk data in `.h5ad` format and run the following command for preprocessing and clustering for selecting the initial batch of representative samples.

```shell
usage: initsetup    [-h] --bulk BulkData --name Name [--normed Normed] 
                    [--geneselection GeneSelection] [--batch BatchSize]

scsemiprofiler initsetup

required arguments:
    --bulk BulkData
                            Input bulk data as a `.h5ad` file. Sample IDs should be 
                            stored in obs.['sample_ids']. Gene symbols should be 
                            stored in var.index. Values should either be raw read 
                            counts or normalized expression.
    --name Name
                            Project name.

optional arguments:
    -h, --help              Show this help message and exit.

    --normed Normed
                            Whether the library size normalization has already been 
                            done (Default: no)

    --geneselection GeneSelection
                            Whether to perform highly variable gene selection: 
                            'yes', 'no', or specify the number of top genes.
                            (Default: yes)

    --batch BatchSize
                            The representative sample batch size
                            (Default: 4)
```
After executing, the preprocessed bulk data and clustering information will be stored automatically. 

**b,** Representative Single-cell Profiling and Processing
This step process the single-cell data (also `.h5ad` format) for the representatives, including the standard single-cell preprocessing and several feature augmentation techniques for enhancing the learning of the deep learning model.Please provide the representatives' single-cell data in the same folder and run the following command.

```shell
usage: scprocess [-h] -singlecell SingleCellData --name Name [--normed Normed] 
                    [--cellfilter CellFilter] [--threshold Threshold] [--geneset 
                    GeneSet] [--weight TopFeatures] [--k K]

scsemiprofiler scprocess

required arguments:
    --singlecell SingleCellData
                            Input new representatives' single-cell data as a `.h5ad` 
                            file. Sample IDs should be stored in obs.['sample_ids']. 
                            Cell IDs should be stored in obs.index. Gene symbols 
                            should be stored in var.index. Values should either be 
                            raw read counts or normalized expression.

    --name Name
                            Project name.

optional arguments:
    -h, --help              Show this help message and exit.

    --normed Normed
                            Whether the library size normalization has already been 
                            done (Default: no)

    --cellfilter CellFilter
                            Whether to perform cell filtering: 'yes' or 'no'.
                            (Default: yes)

    --threshold Threshold
                            The threshold for removing extremely low 
                            expressed background noise, as a proportion of the    
                            library size.
                            (Default: 1e-3)

    --geneset GeneSet
                            Specify the gene set file: 'human', 'mouse', 'none', or 
                            path to the file
                            (Default: 'human')

    --weight TopFeatures
                            The proportion of top highly variable features to 
                            increase importance weight. 
                            (Default: 0.5)

    --k K
                            K-nearest cell neighbors used for cell graph convolution.
                            (Default: 15)
```
The processed single-cell data will be stored automatically in the 'sc_samples' folder. Once finished, the user can proceed to the next step for single-cell inference.



**c,**  Deep Generative Inference
In this step we use deep generative models to infer the single-cell data for non-representative samples using the bulk data and the representatives' single-cell data. The following infereence command can either be excuted in cohort mode for inferring all non-representatives or in single-sample mode for inferring one target sample using one representative. 


```shell
usage: scinfer [-h] -representatives RepresentativesID --name Name [--cluster 
                ClusterLabels] [--targetid TargetID] [--bulktype BulkType] 
                [--lambdad lambdaD] [--pretrain1batch Pretrain1BatchSize] 
                [--pretrain1lr Pretrain1LearningRate] [--pretrain1vae 
                Pretrain1VAEEpochs] [--pretrain1gan Pretrain1GanIterations] 
                [--lambdabulkr lambdaBulkRepresentative] [--pretrain2lr 
                Pretrain2LearningRate] [--pretrain2vae Pretrain2VAEEpochs] 
                [--pretrain2gan Pretrain2GanIterations] [--inferepochs InferEpochs] 
                [--lambdabulkt lambdaBulkTarget] [--inferlr InferLearningRate]

scsemiprofiler scinfer

required arguments:
    --representatives RepresentativesID
                            Either a `.txt` file including all the IDs of the 
                            representatives used in the current round of 
                            semi-profiling when running in cohort mode, or a single 
                            sample ID when running in single-sample mode. 

    --name Name
                            Project name.

optional arguments:
    -h, --help              Show this help message and exit.

    --cluster ClusterLabels
                            A `.txt` file specifying the cluster membership. 
                            Required when running in cohort mode. 

    --targetid TargetID
                            Sample ID of the target sample when running in 
                            single-sample mode.

    --bulktype BulkType
                            Specify 'pseudo' for pseudobulk or 'real' for real bulk data.
                            (Default: real)

    --lambdad lambdaD
                            Scaling factor for the discriminator loss for training the VAE generator.
                            (Default: 4.0)

    --pretrain1batch Pretrain1BatchSize
                            Sample Batch Size of the first pretrain stage.
                            (Default: 128)

    --pretrain1lr Pretrain1LearningRate
                            Learning rate of the first pretrain stage.
                            (Default: 1e-3)

    --pretrain1vae Pretrain1VAEEpochs
                            The number of epochs for training the VAE generator 
                            during the first pretrain stage.
                            (Default: 100)

    --pretrain1gan Pretrain1GanIterations
                            The number of iterations for training the generator and 
                            discriminator jointly during the first pretrain stage.
                            (Default: 100)

    --lambdabulkr lambdaBulkRepresentative
                            Scaling factor for the representative bulk loss.
                            (Default: 1.0)

    --pretrain2lr Pretrain2LearningRate
                            Learning rate of the second pretrain stage.
                            (Default: 1e-4)

    --pretrain2vae Pretrain2VAEEpochs
                            The number of epochs for training the VAE generator 
                            during the second pretrain stage.
                            (Default: 50)

    --pretrain2gan Pretrain2GanIterations
                            The number of iterations for training the generator and 
                            discriminator jointly during the second pretrain stage.
                            (Default: 50)

    --inferepochs InferEpochs
                            The number of epochs for training the generator in each 
                            mini-stage during the inference.
                            (Default: 150)

    --lambdabulkt lambdaBulkTarget
                            Scaling factor for the intial target bulk loss.
                            (Default: 8.0)

    --inferlr InferLearningRate
                            Learning rate during the inference stage.
                            (Default: 2e-4)
```
The inferred data will be stored in the folder "inferreddata" automatically. Once the single-cell inference is finished for all the non-representative samples, you may choose to stop the pipeline and proceed to downstream analyses using the semi-profiled single-cell cohort. You may also proceed to step (**d**) and use active learning to select the next batch of representative samples to further improve the semi-profiling. 


**d,**  Representative Selection Decision

The following command generates the next round of representatives and cluster membership information and store them as `.txt` files in the "status" folder. Then you will provide single-cell data for the new representatives and execute steps (**b**) and (**c**) again to achieve better semi-profiling performance. 

```shell
usage: activeselection [-h] -representatives RepresentativesID --cluster ClusterLabels [--batch Batch] [--lambdasc Lambdasc] [--lambdapb Lambdapb]

scsemiprofiler scprocess

required arguments:
    --representatives RepresentativesID
                            A `.txt` file including all the IDs of the 
                            representatives used in the current round of 
                            semi-profiling.

    --cluster ClusterLabels
                            A `.txt` file specifying the cluster membership. 

    --name Name
                            Project name.

optional arguments:
    -h, --help              Show this help message and exit.

    --batch Batch           The batch size of representative selection
                            (default: 4)

    --lambdasc Lambdasc
                            Scaling factor for the single-cell transformation 
                            difficulty from the representative to the target
                            (Default: 1.0)

    --lambdapb Lambdapb
                            Scaling factor for the pseudobulk data difference
                            (Default: 1.0)
```


**e,** Downstream Analyses
Once the semi-profiling is finished, the semi-profiled data can be used for all single-cell level downstream analysis tasks. We provide some examples in the 'semiresultsanalysis.ipynb' files in each public dataset folder.

## Example
We provide example bulk and single-cell samples in the "example_data" folder. You can use the jupyter notebook 'example.ipynb' to semi-profile a small example cohort and perform some visualization to check the semi-profiling performance. You can expect results similar to the graph below. Based on the representative's cells and bulk difference, the deep generative learning model generates inferred cells for the target sample. The inferred target sample cells have a lot of overlap with the ground truth target sample cells. 
![flowchart](./inference_example.jpg)

<!---You can perform semi-profiling on this example dataset using the following command:   

1. Perform the initial setup and get initial representatives.   
    ```
    initsetup  --bulk example_data/bulkdata.h5ad --name testexample --normed yes] 
                        --geneselection no --batch 2
    ```
2. Get single-cell data for representatives.
    ```
    get_eg_representatives --name testexample 
    ```
3. Process the single-cell data, performing feature augmentations.  
    ```
    scprocess  -singlecell testexample/representative_sc.h5ad --name testexample 
                        --normed yes    --cellfilter no  
    ```
4. Infer the single-cell data for non-representative samples.
    ```
    scinfer   -representatives testexample/status/init_representatives.txt  --name  
                    testexample --cluster testexample/status/init_cluster_labels.txt
    ```

5. Use active learning to select the next round and continue the loop (optional).
    ```
    activeselection -representatives testexample/status/init_representatives.txt 
        --name testexample    --batch 2
        --cluster testexample/status/init_cluster_labels.txt
    ```
-->


## Results reproduction
The three folders correspond to the everything relevant to the three cohorts we used to examine the performance of scSemiProfiler. 


### Raw data availability
The preprocessed COVID-19 dataset is from [Stephenson et al.'s study](https://www.nature.com/articles/s41591-021-01329-2) and can be downloaded from Array Express under accession number [E-MTAB-10026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026) The cancer dataset is from [Joanito et al.'s study](https://www.nature.com/articles/s41588-022-01100-4#Sec2). The count expression matrices are available through Synapse under the accession codes [syn26844071](https://www.synapse.org/#!Synapse:syn26844071/wiki/615389) The iMGL dataset is from [Ramaswami1 et al.'s study](https://www.biorxiv.org/content/10.1101/2023.03.09.531934v1.full.pdf).The raw count iMGL bulk and single-cell data can be downloaded from Gene Expression Omnibus (GEO) repository under accesssion number [GSE226081](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226081).

 
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



