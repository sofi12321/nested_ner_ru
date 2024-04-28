from collections import defaultdict
from typing import Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import json

from utils.instruct_dataset import Instruction
from utils.nerel.nerel_utils import INSTRUCTION_TEXT, ENTITY_TYPES
from utils.instruct_utils import MODEL_INPUT_TEMPLATE, create_output_from_entities



def create_train_test_instruct_datasets(
    data_path: str,
    max_instances: int = -1,
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True,
    test_size: float = 0.3,
    random_seed: int = 42
) -> tuple[list[Instruction], list[Instruction]]:
    
    with open("utils/nerel/train_data.jsonl", "r") as f:
        train_data = json.load(f)
    with open("utils/nerel/val_data.jsonl", "r") as f:
        val_data = json.load(f)

    return train_data, val_data

def create_instruct_dataset(
    data_path: str,
    max_instances: int = -1,
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True
) -> list[Instruction]:
    
    with open("utils/nerel/test_data.jsonl", "r") as f:
        test_data = json.load(f)

    return test_data
