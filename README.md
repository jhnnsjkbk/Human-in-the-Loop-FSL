# Instance Selection Mechanisms for Human-in-the-Loop Systems in Few-Shot Learning
This is a [PyTorch](https://pytorch.org) implementation for Human-in-the-Loop Few-Shot Learning.
We analyze an extensive range of mechanisms that can be used to acquire human expert knowledge for instances that have an uncertain prediction outcome. We show that the acquisition of human expert knowledge significantly accelerates the few-shot model performance given a negligible labeling effort. We validate our findings in various experiments on a benchmark dataset in computer vision and real-world datasets. 

## :bulb: Approach
<img src="/src/images/Approach.png"/>
<img src="/src/images/Approach2.png"/>

## :checkered_flag: Results
<img src="/src/images/Results.png"/>

## :chart_with_upwards_trend: Selection Strategy
<img src="/src/images/SelectionStrategy.png"/>


## :speech_balloon: Setup

### Preliminaries

* Python 3.8
* PyTorch 1.7
* Miniconda3
* Training was executed on a NVIDIA A100 GPU.

Create a [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment by running the following command:

```bash
conda env create -f environment.yml
```

## :speech_balloon: Data

### miniImagenet
Acquire miniImagenet from Ravi & Larochelle '17 and add the file 'images.zip' under datasets/miniImagenet/
https://github.com/floodsung/LearningToCompare_FSL provides a download of miniImagenet.

Then run:
```bash
python src/datasets/create_dataset_miniImagenet.py
```

## :speech_balloon: Citation
If you use our code, please consider citing our work as follows:

```@inproceedings{Jakubik.2022,
  title={Instance Selection Mechanisms for Human-in-the-Loop Systems in Few-Shot Learning},
  booktitle={Proceedings of the International Conference on Wirtschaftsinformatik},
  author={Jakubik, Johannes and Blumenstiel, Benedikt and Voessing, Michael and Hemmer, Patrick},
  year={2022}
  }
``` 
