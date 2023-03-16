# Planet Sentinel Multimodality

<img src="/doc/multimodal_Schematic_representation.jpg" alt="Workflow"
	title="Schematic Represenatation" width="960" height="480" />


## Dataset

The dataset for the experiments can be downloaded from the [link](https://zenodo.org/record/7737587#.ZBHbgNXMJH4)

## Dataset development

In case one wants to setup their own data, they need to download the DENETHOR dataset from the [data portal](https://mlhub.earth/data/dlr_fusion_competition_germany).

Following the downloading of requisite files, the script utils/prepare_data.py for pre-training and data for downstream task1. For downstream task2, the script utils/prepare_val_data.py can be used.

## Installation

Please look into sample batch scripts for modules required and requirements.txt for additional python packages.

## How to Start 

Download the data from the links shared above.

Create a folder in utils name "h5_folder" and put all the files inside the h5_folder


```
git clone <repo>
cd <repo>
cd slurm_scripts
'''
run_main.sbatch for baseline supervised experiments
run_sentinel2_self_supervisd.sbatch for experiments with single models
run_self_supervised.sbatch for experiments with two modes

sbatch <interested_script> #change sbatch directives according to your available resources

```


## Name
Multimodal Contrastive Learning for Boosting Crop Classification Using Sentinel2 and Planetscope

## Description
We have provided three baseline experiments (LSTM, Transfromer and Inceptiontime). We used [breizhcrops](https://github.com/dl4sits/BreizhCrops) for implementation of these networks. These models are implemented in "models/" folder. All the self-supervised models are implemnted in "self_supervised_models/" folder.


## Support
Create issues in case you need support on implementation.

## Contributing
Please send us a PR in case you want to add new methods or find any bugs in our code. You are openly welcomed for that.

## Authors

[Ankit Patnala](https://www.fz-juelich.de/profile/patnala_a)

## License
All the code are open to use and please cite us if you use them.

