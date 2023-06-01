# Speaking the Language of Your Listener

This repository contains the code for the paper "Speaking the Language of Your Listener: Audience-Aware Adaptation via Plug-and-Play Theory of Mind" (to appear in Findings of ACL 2023), where we model a visually grounded referential
game between a knowledgeable speaker and a listener with more limited visual and linguistic experience. We propose an
adaptation mechanism for the speaker, building on plug-and-play approaches to controlled language generation, where
utterance generation is steered on the fly by a simulator without finetuning the speaker's underlying language model.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Repository Structure](#repository-structure)
4. [Experimental Pipeline](#experimental-pipeline)
5. [Citation](#citation)

## Installation

### Setup.py

You can also install by running the following command:

```bash
pip install .
```
the corresponding conda env `uvapb` will be created.

Install nlgeval with:

```bash
pip install git+https://github.com/Maluuba/nlg-eval.git
```

Be sure to comment out [this line](https://github.com/Maluuba/nlg-eval/blob/7f7993035a2f4729a15d20040fd904933ea58767/nlgeval/__init__.py#L289
) in your local installation.

##### Troubleshooting

If you are experiencing a genism error when running the code be sure to have python 3.8 installed.
If you are experiencing a broken pipe error
check [this issue](https://github.com/InnerPeace-Wu/densecap-tensorflow/issues/10) or that you have java installed.
If the error persists disable Metor in the NLGEval class with:

```
 NLGEval( metrics_to_omit=["METEOR"])
```

### Additional Setup

Remember to download spacy with

```bash
python -m spacy download en_core_web_lg
```

## Usage

### Parameters

The parameters are stored in the [Params class](src/commons/Params.py) and can be changed in the class or with argparse.
For example, to change the batch size you can run the following command:

```bash
python train.py --batch_size 64
```

For boolean parameters you can use the following syntax:

```bash
python train.py -use_cuda
```

### Weight and Biases

This project relies heavily on [Weights and Biases](https://wandb.ai/site) for logging and visualization. If you don't
want to use it, you can set `debug` flag to `True` in the [Params class](src/commons/Params.py).

### Training

In this project, there are three agents that need training: the speaker, the listener, and the simulator.

#### Speaker

The speaker is trained with the following command:

```bash
python src/trainers/speaker_train.py --epochs 300
```

#### Listener

The listener is trained with the following command:

```bash
python src/trainers/listener_train.py --epochs 50
```

#### Simulator

The simulator training relies on the speaker and listener models. The simulator is trained with the following command:

```bash
python src/trainers/simulator_pretrain.py --epochs 50
```

### Evaluation

To test the simulator you can run the following command:

```bash
python src/evals/adaptive_speaker.py
```

## Experimental Pipeline

The experimental pipeline of our model includes two agents—a speaker and a listener—implemented as three models: a
generative language model instantiating the speaker, a discriminative model instantiating the listener, and a simulator
used by the speaker to assess the forward effect of its planned utterance. The language model and the discriminator
model are adapted from those by Takmaz et al. (2020), and the simulator model is built on the discriminator’s
architecture with additional components.

Detailed descriptions of the Generative Language Model (Speaker), Discriminator (Listener), and the Simulator can be
found

in the `src/models` directory of the repository.

## Citation

If you find this repository helpful in your research, consider citing our work:

```bib
@misc{takmaz2023speaking,
      title={Speaking the Language of Your Listener: Audience-Aware Adaptation via Plug-and-Play Theory of Mind}, 
      author={Ece Takmaz and Nicolo' Brandizzi and Mario Giulianelli and Sandro Pezzelle and Raquel Fernández},
      year={2023},
      eprint={2305.19933},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
