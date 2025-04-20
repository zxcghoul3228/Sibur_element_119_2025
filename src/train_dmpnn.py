from pathlib import Path
import os
from glob import glob
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
import pandas as pd
import numpy as np
import torch
from chemprop import data, featurizers, models, nn
import chemprop
from rdkit.Chem import MolFromSmiles, MolToSmiles

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['lipo']  # list of names of the columns containing targets



def pretrain_dmpnn(train_path: str = "data/add_data.csv",
                   batch_size: int = 64,
                   max_epochs: int = 50,
                   checkpoint_dir: str = "checkpoint_pretraining",
                   ):
    '''Pretrain D-MPNN model on additional data.'''
    seed_everything(3228, workers=True)
    # Load add data
    add_data = pd.read_csv(train_path)
    add_data = add_data.rename(columns={'LogP': target_columns[0], 'SMILES': 'smiles'})[['smiles', 'lipo']].dropna().reset_index(drop=True)
    smis = add_data.loc[:, smiles_column].values
    ys = add_data.loc[:, target_columns].values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0))
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    # Make graph features
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)    
    # Make dataloaders
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=batch_size, seed=3228)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False, seed=3228)    

    # Make Neural network
    mp = nn.AtomMessagePassing(depth=3)  # Message Passing layers
    agg = nn.NormAggregation()  # Aggregation function
    ffn = nn.RegressionFFN()  # Regression head
    batch_norm = True  # Normalize agg function output
    metric_list = [nn.metrics.RMSE()]  # Validation metric
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    
    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_dir,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=False,
    )
    
    
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,  # number of epochs to train for

        callbacks=[checkpointing],  # Use the configured checkpoint callback
        deterministic=True
    )
    # Train the model
    trainer.fit(mpnn, train_loader, val_loader)
    print("Pretraining validation metric:")
    print(trainer.test(dataloaders=val_loader))


def fit_predict_dmpnn(train_path: str = "./data/prepared_train.csv",
                      val_path=None,
                      test_path: str = "./data/prepared_test.csv",
                      foundation_model: str = "checkpoint_pretraining",
                      n_models: int = 5,
                      batch_size: int = 64,
                      max_epochs: list = [20, 20, 20, 20, 20],
                      checkpoint_dir: str = "checkpoint_finetuned"
                      ) -> list:
    '''Fine-tune D-MPNN model and return ensemble predictions.'''

    # Load add data
    df_input_train = pd.read_csv(train_path, names=['ID', smiles_column, target_columns[0]]).drop(columns='ID', index=0)
    df_input_test = pd.read_csv(test_path, names=['ID', smiles_column]).drop(columns='ID', index=0)
    df_input_test[target_columns[0]] = 0
    smis_train = df_input_train.loc[:, smiles_column].values
    smis_test = df_input_test.loc[:, smiles_column].values
    ys_train = df_input_train.loc[:, target_columns].values
    ys_test = df_input_test.loc[:, target_columns].values
    train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_train, ys_train)]
    test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_test, ys_test)]
    # Make graph features
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data, featurizer)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    # Make dataloaders
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=batch_size, seed=3228)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False, seed=3228)

    # Make Neural network
    test_preds = []
    for i in range(5):
        seed_everything(i, workers=True)
        checkpointing = ModelCheckpoint(
            f"{checkpoint_dir}_{i}",  # Directory where model checkpoints will be saved
            save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        )
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=max_epochs[i],  # number of epochs to train for
            callbacks=[checkpointing],  # Use the configured checkpoint callback
            deterministic=True
        )
        # Load pretained model
        checkpoint_path = glob(f"{foundation_model}/*.ckpt")[0]
        mpnn_cls = models.MPNN
        mpnn = mpnn_cls.load_from_file(checkpoint_path)    
        # Train model
        trainer.fit(mpnn, train_loader)
        # Make prediction
        preds = trainer.predict(dataloaders=test_loader)
        preds_ = []
        for t in preds:
            preds_.extend(t.cpu().numpy().flatten().tolist())
        test_preds.append(preds_)
    return test_preds
