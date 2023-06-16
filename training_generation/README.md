### Data processing
* Sarcasm:
    * Encoded and decoded in unicode.
    

### Data structure
Base dataset: ``data/{construct}/base.json``

Generated dataset: ``{logs}/{task}_{model_name}.json``

### Running the file
To run ```run_inference.py```: "python src/run_inference.py run_name="{task_name}""

### Cleaning outputs of OpenAI text
The texts that OpenAI outputs are often messy. We conduct some simple cleaning in ``src/utils.py``.
* Remove hashtags (OpenAI might add too many) (not convinced by)
* Remove numbers ex. (1) or 2..
* Sometimes the model replies with "Sure, here are ten...". To account for this, we remove all messages where the model
agrees to our task.
*