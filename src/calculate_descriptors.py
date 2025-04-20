import pandas as pd
import numpy as np
import rdkit

from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles, rdMolDescriptors, MACCSkeys, rdFingerprintGenerator
from mordred import Calculator, descriptors
from tqdm import tqdm
from Mold2_pywrapper import Mold2


def calculate_rdkit_descriptors(mols: list) -> pd.DataFrame:
    descs = []
    n = len(mols)
    for mol in tqdm(mols, total=n, desc="Calculate_rdkit_descriptors"):
        desc = list(Descriptors.CalcMolDescriptors(mol).values())
        descs.append(desc)
    cols = list(Descriptors.CalcMolDescriptors(mols[0]).keys())
    descs = np.array(descs)
    #print(descs.shape)
    df = pd.DataFrame(data=descs, columns=cols)
    return df


def calculate_rdkit_properties(mols: list) -> pd.DataFrame:
    properties = rdMolDescriptors.Properties()
    props_names = list(properties.GetPropertyNames())
    props = []
    n = len(mols)
    for mol in tqdm(mols, total=n, desc="Calculate_rdkit_properties"):
        props_ = list(properties.ComputeProperties(mol))
        props.append(props_)
    props = np.array(props)
    #print(descs.shape)
    df = pd.DataFrame(data=props, columns=props_names)
    return df 

def calculate_morgan_fingerprints(mols: list, radius=2, nBits=2048):
    """Генерация фингерпринтов Morgan"""
    #mol = Chem.MolFromSmiles(smiles)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    cols = [f'morgan_{i}' for i in range(nBits)]
    fps = []
    for mol in tqdm(mols, total=len(mols), desc="Calculate_morgan_fingerprints"):
        fp = fp_generator.GetFingerprintAsNumPy(mol)
        fps.append(fp)
    fps = np.array(fps)
    df = pd.DataFrame(data=fps, columns=cols)
    return df


def calculate_maccs_fingerprints(mols: list):
    """Генерация maccs"""
    fps = []
    for mol in tqdm(mols, total=len(mols), desc="Calculate_maccs_fingerprints"):
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
        fps.append(fp)
    cols = [f'maccs_{i}' for i in range(len(fp))]
    fps = np.array(fps)
    df = pd.DataFrame(data=fps, columns=cols)
    return df


def calculate_rdkit_fingerprints(mols: list, fpSize=2048):
    """Генерация rdkit фингерпринтов"""
    fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    fps = []
    for mol in tqdm(mols, total=len(mols), desc="Calculate_rdkit_fingerprints"):
        fp = fp_gen.GetFingerprint(mol).ToList() 
        fps.append(fp)
    cols = [f'rdkit_{i}' for i in range(fpSize)]
    fps = np.array(fps)
    df = pd.DataFrame(data=fps, columns=cols)
    return df


def calculate_mordred_descriptors(mols: list, cols2drop=None):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    mordred_train = calc.pandas(mols)
    df_train = mordred_train.apply(pd.to_numeric, errors='coerce')
    if cols2drop is None:
        cols2drop = df_train.columns[df_train.isna().sum().values > 4000]
    df_train = df_train.drop(columns=cols2drop)
    return df_train, cols2drop


def calculate_mold2_descriptors(mols: list):
    # create descriptor calculator with all descriptors
    mold2 = Mold2()
    train_mold2 = mold2.calculate(mols, show_banner=False)
    return train_mold2


def calculate_descriptors(mols: list,
                          cols2drop=None,
                          # rdkit_desc=True,
                          # rdkit_props=True,
                          # mordred=True,
                          # mold2=True,
                          # morgan_fps=True,
                          # maccs_fps=True,
                          # rdkit_fps=False,
                          ):
    '''Calculate all descriptors.'''
    print("Calculate mordred descriptors")
    df_train, cols2drop = calculate_mordred_descriptors(mols, cols2drop=cols2drop)
    train_rdkit_descriptors = calculate_rdkit_descriptors(mols)
    train_rdkit_properties = calculate_rdkit_properties(mols)


    # Delete duplicate columns
    mordred_cols = df_train.columns
    rdkit_desc_cols = [col for col in train_rdkit_descriptors.columns if col not in train_rdkit_properties.columns]
    rdkit_desc_props_cols = np.concatenate((rdkit_desc_cols, train_rdkit_properties.columns))
    inters = set(mordred_cols).intersection(set(rdkit_desc_props_cols))
    rdkit_desc_props_cols = [col for col in rdkit_desc_props_cols if col not in mordred_cols]
    
    train_rdkit_descriptors_properties = pd.concat((train_rdkit_descriptors[rdkit_desc_cols], train_rdkit_properties), axis=1)
    train_mordred_rdkit_desc_prop = pd.concat((df_train, train_rdkit_descriptors_properties[rdkit_desc_props_cols]), axis=1)

    train_morgan_fps = calculate_morgan_fingerprints(mols)
    train_maccs_fps = calculate_maccs_fingerprints(mols)
    train_rdkit_fps = calculate_rdkit_fingerprints(mols)
    
    train_all_fps = pd.concat((train_morgan_fps, train_maccs_fps, train_rdkit_fps), axis=1)
    
    train_mordred_rdkit_fps = pd.concat((train_mordred_rdkit_desc_prop, train_all_fps), axis=1)
    
    morgan_fp_cols = train_morgan_fps.columns
    maccs_fp_cols = train_maccs_fps.columns
    rdkit_fp_cols = train_rdkit_fps.columns
    print("Calculate mold2 descriptors")
    train_mold2 = calculate_mold2_descriptors(mols)
    train_mordred_rdkit_fps_mold2 = pd.concat((train_mordred_rdkit_fps, train_mold2), axis=1)
    mold2_cols = train_mold2.columns
    
    return train_mordred_rdkit_fps_mold2, cols2drop, mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols, mold2_cols
