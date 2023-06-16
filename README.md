# Sharpness Aware Minimization (SAM) Project

In this project we will explore sharpness aware minimization in the context of synthetic data generation. Past work ([*arXiv*](https://arxiv.org/pdf/2110.08529.pdf)) has suggested that SAM can help with LM generalizability. We extend on this work by synthetically generating data and fine-tuning a model. Synthetic data has been known to have its own artifacts that originate from the LLM. In this paper, we aim to explore if we can use SAM to overcome some of these weaknesses. 

The interest in developing and improving Large Language Models (LLMs) has increased considerably, while new methods such as synthetic data generation came to reinforce the traditional training of these models. With these heavily over-parameterized models, a need for better generalization is present.
A recently proposed optimization algorithm, Sharpness-Aware Minimization (SAM), focusing on loss function convergence to flatter minima, empirically proved to show improved model performance in the computer vision domain. Motivated by these findings, we want to explore whether SAM brings similar improvement when fine-tuning LLMs with synthetic data.

Data generation, fine-tuning the model and evaluation can be found in the folder ``training_generation/``.