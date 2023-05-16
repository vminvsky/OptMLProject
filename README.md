# Sharpness Aware Minimization (SAM) Project

In this project we will explore sharpness aware minimization in the context of synthetic data generation. Past work ([*arXiv*](https://arxiv.org/pdf/2110.08529.pdf)) has suggested that SAM can help with LM generalizability. 

We extend on this work by synthetically generating data and fine-tuning a model. Synthetic data has been known to have its own artifacts that originate from the LLM. In this paper, we aim to explore if we can use SAM to overcome some of these weaknesses. 