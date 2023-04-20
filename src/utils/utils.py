
import yaml
import hydra
import torch
import wandb
import tempfile
import random
import pandas as pd
from collections import defaultdict
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Iterable
from src.utils.types import Result
from pdflatex import PDFLaTeX
from lightning.pytorch import loggers as pl_loggers
from pdfCropMargins import crop


def flatten_tensor_dicts(ds):
    keys = list(ds[0].keys())

    flatten_d = {}
    for key in keys:
        if len(ds[0][key].shape) == 0:
            flatten_d[key] = torch.stack(tuple(d[key] for d in ds))
        elif len(ds[0][key].shape) == 1:
            flatten_d[key] = torch.cat(tuple(d[key] for d in ds))
        else:
            raise ValueError('unsupported tensor shape')


    return flatten_d


def result_to_dataframe(result: Result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result

    return pd.concat(result, ignore_index=True)


def load_dataframes(paths: Iterable[str], contains_index=True) -> Iterable[pd.DataFrame]:
    if contains_index:
        return [
            pd.read_csv(p, index_col=0)
            for p in paths
        ]
    else:
        return [
            pd.read_csv(p)
            for p in paths
        ]


def load_yaml_as_dict(yaml_path: str) -> dict:
    d = None
    with open(yaml_path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    return d

def load_yaml_as_obj(yaml_path: str) -> object:
    d = load_yaml_as_dict(yaml_path)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    obj = Struct(**d)

    return obj


def latex_to_pdf(latex_path: str, pdf_path: str, packages: Iterable[str] = ['booktabs']):
    # loading latex element from a file
    latex_element = ''
    with open(latex_path, 'r') as f:
        latex_element = f.read()

    # composing latex document
    latex_full = ''
    latex_full += r'\documentclass{article}'
    latex_full += r'\pagestyle{empty}'
    latex_full += '\n'.join(r'\usepackage{' +
                            package + r'}' for package in packages)
    latex_full += r'\begin{document}'
    latex_full += latex_element
    latex_full += r'\end{document}'

    # saving the latex document into a temporary file
    tmp = tempfile.NamedTemporaryFile(suffix='.tex')
    with open(tmp.name, 'w') as f:
        f.write(latex_full)

    # compiling a pdf
    with open(tmp.name, 'rb') as f:
        pdfl = PDFLaTeX.from_binarystring(f.read(), 'latex_to_pdf')

    pdf, _, _ = pdfl.create_pdf()

    # saving the pdf into a temporary file
    tmp = tempfile.NamedTemporaryFile(suffix='.pdf')
    with open(tmp.name, 'wb') as f:
        f.write(pdf)

    # cropping the pdf
    crop(['-p', '0', '-a', '-10', '-o', pdf_path, tmp.name])

def have_models_same_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False

    return True


def map_tensor_values(tensor, mapping):
    """
    Map the values of a PyTorch tensor based on a dictionary.

    Args:
        tensor (torch.Tensor): The input tensor.
        mapping (dict): A dictionary that maps input values to output values.

    Returns:
        torch.Tensor: A new tensor with the same shape as the input tensor, where each
        element has been mapped to its corresponding value in the mapping dictionary.
    """

    output_tensor = torch.tensor([mapping[i.item()] for i in tensor])
    output_tensor = to_best_available_device(output_tensor)

    return output_tensor

def recursive_dict_compare(dict1, dict2):
    """
    Recursively compare two dictionaries for equality
    """
    if len(dict1) != len(dict2):
        return False

    for key, value1 in dict1.items():
        value2 = dict2.get(key)
        if isinstance(value1, dict) and isinstance(value2, dict):
            # recurse for nested dictionaries
            if not recursive_dict_compare(value1, value2):
                return False
        else:
            if value1 != value2:
                return False

    return True

# can be a model or a tensor
def to_best_available_device(input):
    if torch.cuda.is_available():
        input = input.to('cuda')
    # elif torch.backends.mps.is_available():
    #     input = input.to('mps')

    return input

def get_the_best_accelerator():
    if torch.cuda.is_available():
        return 'gpu'
    # elif torch.backends.mps.is_available():
    #     return 'mps'

    return 'cpu'

def pick_samples_from_classes_evenly(class_counts, n_total_samples):
    # defining helper variables
    n_classes = len(class_counts)
    class_samples = defaultdict(int)
    class_has_remaining = defaultdict(lambda: True)

    # picking the rest of the samples to reach n_total_samples
    done = False
    while not done:
        elegible_classes = [i for i in range(n_classes) if class_has_remaining[i]]
        random.shuffle(elegible_classes)
        for i in elegible_classes:
            # pick sample from class
            class_samples[i] += 1
            
            # check if they are no remaining samples for that class
            if class_samples[i] == class_counts[i]:
                class_has_remaining[i] = False

            if sum(class_samples.values()) == n_total_samples:
                done = True
                break

    # converting dict to list
    class_samples = [class_samples[i] for i in range(n_classes)]

    return class_samples
