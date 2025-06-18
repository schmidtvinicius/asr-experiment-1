# asr-experiment-1
This repository contains all the source code required for running the experiments related to research question 1 of group 14 for the course (Automatic) Speech Recognition (LET-REMA-LCEX10).

Author: Vinícius Ribeiro Machado Schmidt - s1123702

## Running GenGAN
The files in this repository were sourced from https://github.com/dimitriStoidis/GenGAN. Instructions on how to run the pre-trained GenGAN model on the audio files can be found in the [Demo section](https://github.com/dimitriStoidis/GenGAN?tab=readme-ov-file#demo) of that repository. Make sure you have activated the gengan conda environment before running the files, e.g.:

If it's your first time running the script:
```bash
conda env create -f gengan.yml
```

If you've already created the environment:
```
conda activate gengan
```

## Running utility experiments
In order to run the experiments on assessing GenGAN's utility, first run the code in the [experiments notebook](./experiments.ipynb). After that, you can run the [utility notebook](utility.ipynb). Make sure you activate the nemo environment for running these notebooks, e.g.:

If it's your first time running the script:
```bash
conda env create -f nemo.yml
```

If you've already created the environment:
```
conda activate nemo
```
