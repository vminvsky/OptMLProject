# Sharpness Aware Minimization (SAM) for Synthetic Data Generalizability

## About
In this project we will explore sharpness aware minimization in the context of synthetic data generation. Past work ([*arXiv*](https://arxiv.org/pdf/2110.08529.pdf)) has suggested that SAM can help with LM generalizability. 

We extend on this work by synthetically generating data and fine-tuning a model. Synthetic data has been known to have its own artifacts that originate from the LLM. In this paper, we aim to explore if we can use SAM to overcome some of these weaknesses. 

## Hyperparameter-search

## Generation

## Finetuning

## Getting Started
This repository contains scripts the following code.
1. In `synthetic_generation` we have the function `run_inference.py` that lets you generate new synthetic examples.
2. In `faithful-data-gen` we have included the repository from some of our previous work on finetuning a model for classification. This is moslty used for templates of the config files used for Hydra and data.
3. In `finetuning` we have our SAM implementation and model for finetuning. To finetune a model go to 