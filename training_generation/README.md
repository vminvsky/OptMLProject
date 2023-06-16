

## Data & Generation of synthetic data

We use the sarcasm detection dataset from the ([*SemEval-2022 Task 6*](https://aclanthology.org/2022.semeval-1.111/)). The train set includes over two thousand self-disclosed instances of sarcasm being shared on Twitter.


### Data processing
Data processing on the sarcasm data:
    * Encoded and decoded in unicode.
    * Tokenization, label preprocessing and column formatting.
    * Split the dataset into train, validation, and test (Size of train set by default 0.9).

### Data structure
Base dataset: ``data/{construct}/base.json``

Generated dataset: ``{logs}/{task}_{model_name}.json``

### Running the file
To run ```run_inference.py```: "python src/run_inference.py run_name="{task_name}""

### Cleaning outputs of OpenAI text
The texts that OpenAI outputs are often messy. We conduct some simple cleaning in ``src/utils.py``.
* Remove hashtags (OpenAI might add too many) (not convinced by)
* Remove numbers ex. (1) or 2..
* Sometimes the model replies with "Sure, here are ten...". To account for this, we remove all messages where the model agrees to our task.
