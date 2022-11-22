# Requirements
from .utils import *
import numpy as np
from tqdm import tqdm
import logging as log
import torch


def evaluate_metrics(model, val_dtl, device):
    # Setup
    idx2tag = {v:k for k,v in val_dtl.dataset.tag2idx.items()}
    NER_LABELS, NER_OUTPUT = [], []
    # Create loop with custom metrics
    log.info("Stack predictions:")
    for batch in tqdm(iter(val_dtl)):
        # Get labels
        ner_labels = batch.get('labels').detach().numpy()
        batch = {k:v.to(device) for k,v in batch.items()}
        # Get output
        with torch.no_grad():
            ner_output = model(**batch)
        ner_output = torch.argmax(ner_output, dim=-1).detach().cpu().numpy()
        # Decode NER arrays
        ner_labels = np.vectorize(idx2tag.get)(ner_labels)
        ner_output = np.vectorize(idx2tag.get)(ner_output)
        # Append results
        NER_LABELS.append(ner_labels)
        NER_OUTPUT.append(ner_output)
    # Build final objects
    NER_LABELS = np.concatenate(NER_LABELS)
    NER_OUTPUT = np.concatenate(NER_OUTPUT)
    # Compute global metrics
    log.info("Compute global metrics:")
    f1NER, precision, recall = computeF1Score(NER_OUTPUT, NER_LABELS)
    return {'f1_NER':f1NER,'precision_NER':precision,'recall_NER':recall}