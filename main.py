# Requierments
import logging as log
import json
import os
import shutil
import sys
import torch
import wandb
import fire

# Dependencies
from src.setup import setup_data
from src.dataset import NER_Dataset
from src.fitter import NER_Fitter
from src.model import NER_Model
from src.loss import FocalLoss
from src.callbacks import wandb_checkpoint
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
    train_dtl = torch.utils.data.DataLoader(train_dts,
                                            batch_size=train_dct['batch_size'],
                                            num_workers=train_dct['num_workers'],
                                            shuffle=True,
                                            )
    val_dts = NER_Dataset(test_annot, train_dct['HuggingFace_model'], max_length, tag2idx)
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=2*train_dct['batch_size'],
                                          num_workers=train_dct['num_workers'],
                                          shuffle=False,
                                          )
    # Define model
    log.debug(f"Get model:")
    model = NER_Model(train_dct['HuggingFace_model'], num_labels, train_dct['dropout'], train_dct['device'])

    #
    # Part III: Prepare Trainer
    #
    
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Get loss, optimisers and schedulers
    log.info(f"Prepare model, loss function, optimizer and scheduler")
    model = NER_Model(
        train_dct['HuggingFace_model'],
        num_labels,
        train_dct['dropout'],
        train_dct['device']
    )
    model = torch.compile(model) # PyTorch 2.0
    criterion = torch.nn.CrossEntropyLoss(
        reduction='mean'
    )
    #criterion = FocalLoss(
    #    gamma=train_dct['gamma_loss'],
    #    temperature=train_dct['temperature_loss'],
    #    from_logits=True,
    #    multilabel=False,
    #    reduction='mean',
    #    n_classes=num_labels,
    #    class_weights=None,
    #    device=train_dct['device']
    #)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=train_dct['learning_rate'],
                                  weight_decay=train_dct['weight_decay'],
                                  )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=train_dct['learning_rate'],
                                                    steps_per_epoch=len(train_dtl),
                                                    epochs=train_dct['epochs'],
                                                    pct_start=train_dct['warmup_epochs_factor'],
                                                    anneal_strategy='cos',
                                                    )

    #
    # Part IV: Train model
    #

    # Trainer
    log.debug(f"Start model training:")
    if not os.path.isdir(os.path.join(os.getcwd(),train_dct['filepath'])): os.makedirs(os.path.join(os.getcwd(),train_dct['filepath']))
    fitter = NER_Fitter(
        model,
        train_dct['device'],
        criterion,
        optimizer,
        scheduler,
        step_scheduler=True,
        validation_scheduler=False,
        folder=os.path.join(os.getcwd(),train_dct['filepath']),
        use_amp = bool(train_dct['use_amp'])
    )
    # Weights and Biases session
    wandb.login(key=wandb_dct['WB_KEY'])
    wandb.init(project=wandb_dct['WB_PROJECT'], entity=wandb_dct['WB_ENTITY'], config=train_dct)
    # Training
    log.info(f"Start fitter training:")
    _ = fitter.fit(train_loader = train_dtl,
                   val_loader = val_dtl,
                   n_epochs = train_dct['epochs'],
                   metrics = None,
                   early_stopping = train_dct['early_stopping'],
                   early_stopping_mode = train_dct['scheduler_mode'],
                   verbose_steps = train_dct['verbose_steps'],
                   step_callbacks = [wandb_checkpoint],
                   validation_callbacks = [wandb_checkpoint],
                   )
    # Remove objects from memory
    del fitter, criterion, optimizer, scheduler, train_dts, train_dtl
    torch.cuda.empty_cache()
    
    #
    # Part V: Evaluation
    #

    # Load best checkpoint
    log.info("Loading best model checkpoint:")
    ckpt = torch.load(os.path.join(os.getcwd(),train_dct['filepath'],'best-checkpoint.bin'))
    model = NER_Model(
        train_dct['HuggingFace_model'],
        num_labels,
        train_dct['dropout'],
        train_dct['device']
    )
    model.load_state_dict(ckpt['model_state_dict'])

    # Calculate metrics
    log.debug("Compute metrics on evaluation dataset:")
    metrics_dct = evaluate_metrics(model, val_dtl, train_dct['device'])
    wandb.log({'Global metrics':wandb.Table(data=[list(metrics_dct.values())], columns=list(metrics_dct.keys()))})

    # End WB session
    log.debug(f"End Weights and Biases session:")
    last_model=max([int(elem.split('-')[-1]) for elem in os.listdir(os.path.join(os.getcwd(),'output'))])
    shutil.move(os.path.join(os.getcwd(),'output',f"checkpoint-{last_model}"), os.path.join(f"{wandb.run.dir}",f"checkpoint-{last_model}"))
    wandb.finish()
   

if __name__=="__main__":
    fire.Fire(main)