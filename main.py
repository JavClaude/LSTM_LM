import os
import json
import logging
import argparse
import configparser

import tqdm
import torch
import mlflow
import tokenizers

from Model.model import LSTMModel
from Tokenizer.train_tokenizer import train_trokenizer
from Training.training_eval import train_model, eval_model
from Utils.data import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read("config.cfg")

os.environ["MLFLOW_TRACKING_URI"] = config["mlflow"]["MLFLOW_TRACKING_URI"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["mlflow"]["MLFLOW_S3_ENDPOINT_URL"]

os.environ["AWS_ACCESS_KEY_ID"] = config["aws-credentials"]["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["aws-credentials"]["AWS_SECRET_ACCESS_KEY"]

## Mlflow configuration ##

mlflow.set_experiment(config["experiment"]["experiment_name"])
artifact_path = config["experiment"]["artifact_path"]

def main(**kwargs) -> None:
    if os.path.isdir("tmp"):
        pass
    else:
        os.mkdir("tmp")

    if kwargs.get("path_to_tokenizer") is None:
        tokenizer = train_trokenizer(**{
            "path_to_textfile": kwargs.get("path_to_data_train"),
            "num_merges": kwargs.get("num_merges")
        })
    else:
        tokenizer = tokenizers.Tokenizer.from_file(kwargs.get("path_to_tokenizer"))

    tokenizer.save("tmp/tokenizer.json")
    mlflow.log_artifact("tmp/tokenizer.json", artifact_path="materials")

    kwargs["vocab_size"] = tokenizer.get_vocab_size()
    kwargs.pop("num_merges") 

    trainDataset = TextDataset(kwargs.get("path_to_data_train"), tokenizer, kwargs.get("bptt"), kwargs.get("batch_size"))
    testDataset = TextDataset(kwargs.get("path_to_data_test"), tokenizer, kwargs.get("bptt"), kwargs.get("batch_size"))

    Model = LSTMModel(**kwargs)
    Criterion = torch.nn.CrossEntropyLoss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=kwargs.get("lr"))#), momentum=0.9, nesterov=True)

    kwargs['Optimizer'] = str(Optimizer.__class__).split(".")[-1].replace("'>", "")
    if kwargs['Optimizer'] == "SGD":
        kwargs['Nesterov'] = True

    Model.zero_grad()
    Model.to(device)

    logger.info("Start training for: {}".format(kwargs.get("epochs")))

    mlflow.log_params(kwargs)

    global_train_it = 0
    global_eval_it = 0

    for _ in tqdm.tqdm(range(kwargs.get("epochs"))):
        _, global_train_it = train_model(
            Model,
            trainDataset,
            Criterion,
            Optimizer,
            kwargs["clip_grad_norm"],
            global_train_it
        )

        eval_loss, global_eval_it = eval_model(
            Model,
            testDataset,
            Criterion,
            global_eval_it
        )


    with open("tmp/config_file.json", "w") as file:
        json.dump(kwargs, file)
    
    mlflow.log_artifact("tmp/config_file.json", artifact_path="materials")

    mlflow.pytorch.log_model(Model, artifact_path="materials", code_paths=["Model"])



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_data_train", type=str, required=True)
    argument_parser.add_argument("--path_to_data_test", type=str, required=True)
    argument_parser.add_argument("--path_to_tokenizer", type=str, required=False)
    argument_parser.add_argument("--num_merges", type=int, required=False, default=40000)
    argument_parser.add_argument("--epochs", type=int, required=False, default=10)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=128)
    argument_parser.add_argument("--bptt", type=int, required=False, default=64)
    argument_parser.add_argument("--lr", type=float, required=False, default=0.006)
    argument_parser.add_argument("--momentum", type=float, required=False, default=0.9)
    argument_parser.add_argument("--clip_grad_norm", type=float, required=False, default=0.5)
    argument_parser.add_argument("--embedding_dim", type=int, required=False, default=300)
    argument_parser.add_argument("--hidden_units", type=int, required=False, default=256)
    argument_parser.add_argument("--n_layers", type=int, required=False, default=3)
    argument_parser.add_argument("--bidirectional", type=bool, required=False, default=False)
    argument_parser.add_argument("--dropout_rnn", type=float, required=False, default=0.2)
    argument_parser.add_argument("--dropout", type=float, required=False, default=0.3)

    arguments = argument_parser.parse_args()

    main(**vars(arguments))
