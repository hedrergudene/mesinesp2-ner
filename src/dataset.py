# Requirements
import torch
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer

# Dataset
class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 annotations,
                 model_name:str,
                 max_length:int,
                 tag2idx:Dict,
                 ):
        # Parameters
        self.annotations = annotations
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Fetch text
        annot = self.annotations[idx]
        # Get tokens, attention mask and NER labels
        return self._collate_HuggingFace(annot)

    def _collate_HuggingFace(self, annotation):
        # Tokenize text
        tokens = self.tokenizer.encode_plus(annotation['text'],
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_offsets_mapping=True,
                                            return_tensors='pt',
                                            )
    
        # Create array to store each class labels
        ## First axis indicates the label
        ## Second axis each text
        ## Third axis the token position
        targets = np.zeros((self.max_length), dtype='int32') #Everything is unlabelled by Default
    
        # FIND TARGETS IN TEXT AND SAVE IN TARGET ARRAYS
        offsets = np.squeeze(tokens['offset_mapping'].numpy())
        offset_index = 0
        for _, (start, end, label) in enumerate(annotation['entities']):
            a = int(start)
            b = int(end)
            if offset_index>len(offsets)-1:
                break
            c = offsets[offset_index][0] # Token start
            d = offsets[offset_index][1] # Token end
            count_token = 0 # token counter
            beginning = True
            while b>c: # While tokens lie in the discourse of a specific entity
                if (c>=a)&(b>=d): # If token is inside discourse
                    if beginning:
                        targets[offset_index] = self.tag2idx['B-'+label]
                        beginning = False
                    else:
                        targets[offset_index] = self.tag2idx['I-'+label]
                count_token += 1
                offset_index += 1 # Move to the next token
                if offset_index>len(offsets)-1: # If next token is out of this entity range, jump to the next entity
                    break
                c = offsets[offset_index][0]
                d = offsets[offset_index][1]
        # Save in dictionary
        ner_target =  torch.LongTensor(targets)
        # End of method
        return {'input_ids': torch.squeeze(tokens['input_ids']), 'attention_mask':torch.squeeze(tokens['attention_mask']), 'labels':ner_target}