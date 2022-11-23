# Requirements
import os
import requests
import itertools
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer
import logging as log
from typing import Dict
from .utils import ensemble_corpus_annotations

# Method to load data and compute 
def setup_data(
    train_dct:Dict,
    annotations_train_path:str='input/mesinesp2_train_annotations.json',
    annotations_test_path:str='input/mesinesp2_test_annotations.json'
):

    if (train_dct['train_data_path']!='') & (train_dct['val_data_path']!=''):
        #
        # Part I: Data gathering
        #

        # Generate input directory and download main data
        response = requests.get('https://zenodo.org/record/5602914/files/Subtrack1-Scientific_Literature.zip?download=1', stream=True)
        if response.status_code == 200:
            with open('input/Subtrack1-Scientific_Literature.zip', 'wb') as f:
                f.write(response.raw.read())
        # Download additional data
        response = requests.get('https://zenodo.org/record/5602914/files/Additional%20data.zip?download=1', stream=True)
        if response.status_code == 200:
            with open('input/Additional_Data.zip', 'wb') as f:
                f.write(response.raw.read())
        # Data extraction
        for zipfile_name in [elem for elem in os.listdir("input") if elem[-3:]=='zip']:
            with ZipFile("input/"+zipfile_name) as zf:
                zf.extractall("input/")
            os.remove("input/"+zipfile_name)
        # Build tag2idx dictionary following IB schema
        tag2idx = {
            **{schema+tag:idx for (tag, schema),idx in zip(itertools.product(['disease', 'medication', 'procedure', 'symptom'], ['B-','I-']), range(1,9))},
            **{'O':0}
        }
        # Get training-test annotations
        annot = ensemble_corpus_annotations(
            "input/Subtrack1-Scientific_Literature/Train/training_set_subtrack1_all.json",
            "input/Additional data/Subtrack1-Scientific_Literature/entities_subtrack1_training.json"
        )
        annot_df = pd.DataFrame(annot)
        annot_df['len'] = annot_df['entities'].apply(lambda x: len(x) if len(x)<=40 else 40)
        _, _, train_idx, test_idx = train_test_split(annot_df['text'], range(len(annot_df)), stratify=annot_df['len'])
        train_annot = [x for idx, x in enumerate(annot) if idx in train_idx]
        test_annot = [x for idx, x in enumerate(annot) if idx in test_idx]
        # Save results
        with open(annotations_train_path, 'w') as fout:
            json.dump(train_annot, fout)
        with open(annotations_test_path, 'w') as fout:
            json.dump(test_annot, fout)
    else:
        with open(train_dct['train_data_path'], 'r') as fout:
            train_annot = json.load(fout)
        with open(train_dct['val_data_path'], 'r') as fout:
            test_annot = json.load(fout)        

    #
    # Part II: Tokenizer and data insights
    #

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_dct['HuggingFace_model'])
    # Data insights
    num_labels = len(tag2idx)
    # Estimate max length on training data
    length_list = []
    for elem in tqdm(train_annot):
        length_list.append(len(tokenizer(elem['text']).input_ids))
    max_length = int(np.quantile(length_list, .995))
    log.info(f"Recommended maximum length: {max_length}")

    # Exit
    return tag2idx, max_length, num_labels