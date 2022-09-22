# Instance Selection Mechanisms for Human-in-the-Loop Systems in Few-Shot Learning

## Setup

Create a [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment by running the following command:

```bash
conda env create -f environment.yml
```

## Data

### miniImagenet
Acquire miniImagenet from Ravi & Larochelle '17 and add the file 'images.zip' under datasets/miniImagenet/
https://github.com/floodsung/LearningToCompare_FSL provides a download of miniImagenet.

Then run:
```bash
python src/datasets/create_dataset_miniImagenet.py
```

## Citation
If you use our code, please consider citing our work as follows:

```@inproceedings{Jakubik.2022,
  title={Instance Selection Mechanisms for Human-in-the-Loop Systems in Few-Shot Learning},
  booktitle={Wirtschaftsinformatik},
  author={Jakubik, Johannes and Blumenstiel, Benedikt and Voessing, Michael and Hemmer, Patrick},
  year={2022}
  }
``` 
