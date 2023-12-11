# subtask_a_monolingual
The model is a CNN layer that takes the output of RoBERTa embeddings to classify a piece of text as machine generated or human generated.
## Installation
run the following command:
```bash
pip install -r requirements.txt
```
## Running the code
```python
python cnn_roberta.py --training_data=path/to/train/data.jsonl --dev_data=path/to/dev/data.jsonl --output_file=path/to/output.jsonl
```
