import os
import pandas as pd
import numpy as np

from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles, rdMolDescriptors, MACCSkeys, rdFingerprintGenerator
from chython import smiles
from collections import defaultdict


import warnings
warnings.filterwarnings("ignore")


from calculate_descriptors import calculate_descriptors
from train_classic_ml import train_catboost

def smile_mol(smis: str) -> chython.containers.molecule.MoleculeContainer:
    '''Parse smiles string.'''
    try:
        res = smiles(smis)
        return res
    except:
        return
    
    
def standardize(mol_list: list) -> None:
    '''Standardize list of molecules.'''
    for m in mol_list:
        try:
            m.clean_stereo()
            m.canonicalize()
        except:
            pass


def sanitize_smiles(smis):
    '''Canonize smiles.'''
    try:
        return MolToSmiles(MolFromSmiles(smis))
    except:
        return
    

def prepare_train_test_data(train_path: str, test_path: str):
    '''Prepare input data.
    Args:
    train_path (str): path to train dataset.
    test_path (str): path to test dataset.
    '''

    # Read csv-files
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    # Extract smiles
    test_smiles_list = test_data['SMILES'].tolist()
    train_smiles_list = train_data['SMILES'].tolist()
    # Parse smiles
    train_mols = [smile_mol(m) for m in train_smiles_list]
    test_mols = [smile_mol(m) for m in test_smiles_list]
    # Standardize smiles
    standardize(train_mols)
    standardize(test_mols)
    # Convert to rdkit.Mol
    train_rdkit_mols = [MolFromSmiles(str(m)) for m in train_mols]
    test_rdkit_mols = [MolFromSmiles(str(m)) for m in test_mols]
    train_data['molecule'] = train_rdkit_mols
    test_data['molecule'] = test_rdkit_mols
    # Drop incorrect smiles
    train_data = train_data.loc[~train_data['molecule'].isna()]
    # Cononize smiles
    train_data["SMILES"] = train_data["molecule"].apply(MolToSmiles)
    test_data["SMILES"] = test_data["molecule"].apply(MolToSmiles)
    # Extract duplicates
    train_data_wo_duplicates = train_data.loc[~train_data.duplicated(subset='SMILES', keep=False)]
    train_data_duplicates = train_data.loc[train_data.duplicated(subset='SMILES', keep=False)]

    train_rdkit_mols = train_data_wo_duplicates.molecule.to_list()
    test_rdkit_mols = test_data.molecule.to_list()
    train_dupl_rdkit_mols = train_data_duplicates.molecule.to_list()
    Y_train = train_data_wo_duplicates['LogP']
    # Calculate all descriptors
    train_mordred_rdkit_fps_mold2, cols2drop, mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols, mold2_cols = calculate_descriptors(train_rdkit_mols)
    test_mordred_rdkit_fps_mold2, *_ = calculate_descriptors(test_rdkit_mols, cols2drop=cols2drop)
    train_dupl_mordred_rdkit_fps_mold2, *_ = calculate_descriptors(train_dupl_rdkit_mols, cols2drop=cols2drop)
    # Train model to fix duplicates
    model = train_catboost(train_mordred_rdkit_fps_mold2[np.concatenate((mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols))],
                           Y_train,
                           kwargs={'iterations': 2172, 'learning_rate': 0.027336, 'verbose': 0})
    # Choose target value nearest to model prediction
    preds_duplicates = model.predict(train_dupl_mordred_rdkit_fps_mold2[np.concatenate((mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols))])
    train_data_duplicates['LogP_pred'] = preds_duplicates
    train_data_duplicates['MAE'] = np.abs(train_data_duplicates['LogP'] - train_data_duplicates["LogP_pred"])
    train_data_duplicates['min_mae'] = train_data_duplicates.groupby('SMILES')["MAE"].transform(min)
    train_data_duplicates = train_data_duplicates.reset_index(drop=True)
    train_data_add = train_data_duplicates.loc[train_data_duplicates['MAE'] == train_data_duplicates["min_mae"], ["ID", "SMILES", "LogP", "molecule"]]
    # Deduplicated train data
    train_data_deduplicated = pd.concat((train_data_wo_duplicates, train_data_add)).reset_index(drop=True)
    train_mordred_rdkit_fps_mold2_deduplicated = pd.concat((train_mordred_rdkit_fps_mold2, train_dupl_mordred_rdkit_fps_mold2.loc[train_data_duplicates['MAE'] == train_data_duplicates["min_mae"]])).reset_index(drop=True)
    # Remove outliers
    train_data_deduplicated_wo_outliers = train_data_deduplicated.loc[(train_data_deduplicated['LogP'] < 10) & (train_data_deduplicated['LogP'] > -2)].reset_index(drop=True)
    train_mordred_rdkit_fps_mold2_deduplicated_wo_outliers = train_mordred_rdkit_fps_mold2_deduplicated.loc[(train_data_deduplicated['LogP'] < 10) & (train_data_deduplicated['LogP'] > -2)].reset_index(drop=True)
    # Save prepared datasets and descriptors dataframes
    os.makedirs("./data", exist_ok=True)
    train_data_deduplicated_wo_outliers = train_data_deduplicated_wo_outliers.drop(columns='molecule')
    test_data = test_data.drop(columns='molecule')
    train_data_deduplicated_wo_outliers.to_csv("./data/prepared_train.csv", index=False)
    train_mordred_rdkit_fps_mold2_deduplicated_wo_outliers.to_csv("./data/descriptors_train.csv", index=False)
    test_data.to_csv("./data/prepared_test.csv", index=False)
    test_mordred_rdkit_fps_mold2.to_csv("./data/descriptors_test.csv", index=False)

    return (train_data_deduplicated_wo_outliers,
            train_mordred_rdkit_fps_mold2_deduplicated_wo_outliers,
            test_data,
            test_mordred_rdkit_fps_mold2,
            mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols, mold2_cols)

def prepare_add_data(add_data_path: str, test_data):
    '''Prepare file with additional data.'''
    add_data = pd.read_csv("alogps_3_01_training_.csv")
    add_data['logPow'] = add_data['logPow'].apply(pd.to_numeric, errors='coerce')
    add_data = add_data.rename(columns={'logPow': 'LogP'})[['SMILES', 'LogP']].dropna()
    add_data['ID'] = -1
    add_data['SMILES'] = add_data["SMILES"].apply(sanitize_smiles)
    add_data = add_data.dropna()    
    print("Len add data", len(add_data))
    # Intersection with test data
    inters = (set(test_data.SMILES.values).intersection(set(add_data.SMILES.values)))
    # Save add data
    os.makedirs("./data", exist_ok=True)
    add_data.loc[~add_data["SMILES"].isin(inters)].to_csv("data/add_data.csv")  # Remove the molecules from add data included in the test set to prevent data leakage

