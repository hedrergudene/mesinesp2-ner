# Requierments
import logging as log
import json
import os
import shutil
import sys
import torch
import wandb
import fire
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification

# Dependencies
from src.setup import setup_data
from src.dataset import NER_Dataset
#from src.model import NER_Model
from src.loss import FocalLoss
from src.metrics import evaluate_metrics
from src.utils import seed_everything

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
        train_annot_path:str='input/mesinesp2_train_annotations.json',
        test_annot_path:str='input/mesinesp2_test_annotations.json',
        training_config_path:str="input/training_config.json",
        wandb_config_path:str="input/wandb_config.json",
        ):

    #
    # Part I: Read configuration files
    #
    
    #Training
    with open(training_config_path) as f:
        train_dct = json.load(f)
        seed_everything(train_dct['seed'])
    #Wandb
    with open(wandb_config_path) as f:
        wandb_dct = json.load(f)
        os.environ['WANDB_API_KEY'] = wandb_dct['WB_KEY']
        os.environ['WANDB_USERNAME'] = wandb_dct['WB_ENTITY']
        os.environ['WANDB_PROJECT'] = wandb_dct['WB_PROJECT']

    #
    # Part II: Setup data and model
    #

    # Get tools
    log.debug(f"Setup tools:")
    tag2idx, max_length, num_labels = setup_data(train_dct)
    train_dct['max_length'] = max_length
    # Read data
    with open(train_annot_path, 'r') as f:
        train_annot = json.load(f)
    with open(test_annot_path, 'r') as f:
        test_annot = json.load(f)
    # Build datasets
    log.debug(f"Prepare datasets:")
    train_dts = NER_Dataset(train_annot, train_dct['HuggingFace_model'], max_length, tag2idx)
    val_dts = NER_Dataset(test_annot, train_dct['HuggingFace_model'], max_length, tag2idx)
    # Define model
    log.debug(f"Get model:")
    # Update 21/11/2022: Use HF AutoClass to enable the usage of Optimum library
    #model = NER_Model(train_dct['HuggingFace_model'], num_labels, train_dct['dropout'], train_dct['device'])
    model = AutoModelForTokenClassification.from_pretrained(train_dct['HuggingFace_model'], num_labels=num_labels)

    #
    # Part III: Prepare Trainer
    #
    
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Set up arguments
    log.debug(f"Prepare training arguments:")
    steps_per_epoch = len(train_annot)/(train_dct['batch_size']*train_dct['gradient_accumulation_steps'])
    logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
    logging_steps = logging_steps//train_dct['evaluation_steps_per_epoch']
    # Training arguments
    # Check https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
    # to have a deeper understanding of training wrapper hyperparameters
    training_args = TrainingArguments(
        output_dir=os.path.join(os.getcwd(),train_dct['filepath']),
        gradient_accumulation_steps=train_dct['gradient_accumulation_steps'],
        gradient_checkpointing=bool(train_dct['gradient_checkpointing']),
        warmup_steps=logging_steps*train_dct['warmup_steps_factor'],
        learning_rate=train_dct['learning_rate'],
        weight_decay=train_dct['weight_decay'],
        per_device_train_batch_size=train_dct['batch_size'],
        per_device_eval_batch_size=train_dct['batch_size'],
        dataloader_num_workers = train_dct['dataloader_num_workers'],
        num_train_epochs=train_dct['epochs'],
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=logging_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="wandb",  # enable logging to W&B
        run_name=wandb_dct['WB_RUN_NAME'],
        seed=train_dct['seed'],
        fp16=bool(train_dct['fp16'])
    )
    # Trainer
    log.debug(f"Initialise HuggingFace Trainer:")
    # Loss function
    loss_fn = FocalLoss(gamma = train_dct['gamma'], n_classes = model.num_labels)
    # Training loop wrapper
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            ner_labels = inputs.get('labels')
            # forward pass
            outputs = model(inputs.get('input_ids'), inputs.get('attention_mask'))
            # compute custom loss
            ner_loss = loss_fn(outputs, ner_labels)
            return (ner_loss, outputs) if return_outputs else ner_loss
    # Initialise training loop object
    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_dts,
        eval_dataset=val_dts,
    )

    #
    # Part IV: Train model
    #

    # Trainer
    log.debug(f"Start model training:")
    trainer.train()

    #
    # Part V: Evaluation
    #

    # Prepare evaluation DataLoader and language list
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=trainer.args.per_device_eval_batch_size,
                                          num_workers=trainer.args.dataloader_num_workers,
                                          shuffle=False, # Important to be aligned with lang list
                                          )

    # Calculate metrics
    log.debug("Compute metrics on evaluation dataset:")
    metrics_dct = evaluate_metrics(trainer, val_dtl)
    wandb.log({'Global metrics':wandb.Table(data=[list(metrics_dct.values())], columns=list(metrics_dct.keys()))})

    # End WB session
    log.debug(f"End Weights and Biases session:")
    last_model=max([int(elem.split('-')[-1]) for elem in os.listdir(os.path.join(os.getcwd(),'output'))])
    shutil.move(os.path.join(os.getcwd(),'output',f"checkpoint-{last_model}"), os.path.join(f"{wandb.run.dir}",f"checkpoint-{last_model}"))
    wandb.finish()

    #
    # Part VI: Optimisation
    #

    # Check: https://huggingface.co/blog/optimum-inference
   

if __name__=="__main__":
    fire.Fire(main)