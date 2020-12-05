import os
import itertools
import logging
import tqdm

import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TextDataset():
    def __init__(self,
                path_to_txt: str,
                tokenizer,
                bptt,
                batch_size: int) -> None:
        
        self.bptt = bptt
        self.batch_size = batch_size

        texts_ids = []
        with open(path_to_txt, "r") as file:
            for text in tqdm.tqdm(file.readlines(), desc="Reading & tokenize file: {}".format(path_to_txt)):
                texts_ids.append(
                    tokenizer.encode("<SOS> " + text + " <EOS>").ids
                )
        
        flat_texts_ids = list(itertools.chain.from_iterable((texts_ids)))    
        self.n_batches = len(flat_texts_ids) // (batch_size*bptt)
        self.texts_ids = np.reshape(flat_texts_ids[0: self.n_batches*batch_size*bptt], (batch_size, -1))

        self.target_texts_ids = np.zeros_like(self.texts_ids)
        self.target_texts_ids[:, :-1] = self.texts_ids[:, 1:]
        self.target_texts_ids[:, -1] = self.texts_ids[:, 0]
    
    def __len__(self) -> int:
        return self.texts_ids.shape[-1]
    
    def get_batches(self, i) -> tuple:
        return (
            torch.tensor(self.texts_ids[:, i:i+self.bptt], dtype=torch.long, device=device),
            torch.tensor(self.target_texts_ids[:, i:i+self.bptt], dtype=torch.long, device=device)
        )
        