# Language-Model

In this repository you will find all the ressources for:
1. Train a BPE Tokenizer
2. Train a Word Level LSTM Language Model

Command to run training

```shell
python main.py --path_to_data_train data_train.txt
               --path_to_data_test data_test.txt
               --num_merges 30000 
               --bptt 128
               --lr 0.001
```
