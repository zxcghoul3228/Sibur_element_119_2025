import os
import sys
import pandas as pd
import numpy as np
import argparse
import time
import torch

from src.prepare_data import prepare_train_test_data, prepare_add_data
from src.train_classic_ml import train_catboost, train_lgb
from src.train_dmpnn import pretrain_dmpnn, fit_predict_dmpnn

def main():
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="./data/sibur_element_119_final_train_data80.csv",
        type=str,
        help="Path to train csv-file",
    )
    parser.add_argument(
        "--test_path",
        default="./data/sibur_element_119_final_test_data80.csv",
        type=str,
        help="Path to test csv-file",
    )
    parser.add_argument(
        "--add_data_path",
        default="./data/alogps_3_01_training_.csv",
        type=str,
        help="Path to additional data csv-file",
    )
    parser.add_argument(
        "--alogps_path",
        default="./data/",
        type=str,
        help="Path to directory with ALOGPS descriptors files",
    )
    args = parser.parse_args(sys.argv[1:])
    
    train_path = args.train_path
    test_path = args.test_path
    add_data_path = args.add_data_path
    alogps_path = args.alogps_path
    # Prepare data
    print("Start data preparing")
    (train_data_deduplicated_wo_outliers,
    train_mordred_rdkit_fps_mold2_deduplicated_wo_outliers,
    test_data,
    test_mordred_rdkit_fps_mold2,
    mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols, rdkit_fp_cols, mold2_cols) = prepare_train_test_data(train_path, test_path)

    Y_train = train_data_deduplicated_wo_outliers['LogP']
    # Add ALOGPS descriptors
    train_alogps = pd.read_table(os.path.join(alogps_path, "train_alogps.txt"))
    train_alogps = train_alogps.reset_index().rename(columns={"index": "SMILES", "smiles": "LogP", "logP": "LogS", "logS": "skip"})[["SMILES", "LogP", "LogS"]]
    X_tr = train_mordred_rdkit_fps_mold2_deduplicated_wo_outliers[np.concatenate((mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols))]
    X_tr["LogP_alogps"] = train_alogps["LogP"]
    X_tr["LogS_alogps"] = train_alogps["LogS"]
    test_alogps = pd.read_table(os.path.join(alogps_path, "test_alogps.txt"))
    test_alogps = test_alogps.reset_index().rename(columns={"index": "SMILES", "smiles": "LogP", "logP": "LogS", "logS": "skip"})[["SMILES", "LogP", "LogS"]]
    X_test = test_mordred_rdkit_fps_mold2[np.concatenate((mordred_cols, rdkit_desc_props_cols, morgan_fp_cols, maccs_fp_cols))]
    X_test["LogP_alogps"] = test_alogps["LogP"]
    X_test["LogS_alogps"] = test_alogps["LogS"]
    print("Finish data preparing")
    # Train catboost model
    print("Start CatBoost training")
    model1 = train_catboost(X_tr,
                           Y_train,
                           kwargs={'iterations': 1924, 'learning_rate': 0.05, 'per_float_feature_quantization': ['3855:border_count=1024'], 'verbose': 0
                                   })
    print("Finish CatBoost training")
    print("Start LightGBM training")
    model2 = train_lgb(X_tr,
                       Y_train
                       )
    print("Finish LightGBM training")
    # Make catboost_predictions
    cb_predictions = model1.predict(X_test)
    submission_cb = test_data.drop(columns=['SMILES'])
    submission_cb['LogP'] = cb_predictions
    submission_cb.to_csv("submission_cb.csv", index=False)
    # Make lightgbm_predictions
    lgb_predictions = model2.predict(X_test)
    submission_lgb = test_data.drop(columns=['SMILES'])
    submission_lgb['LogP'] = cb_predictions
    submission_lgb.to_csv("submission_lgb.csv", index=False)
    # CatBoost and LightGBM ensemble
    submission_cb['LogP'] = (submission_cb['LogP'] + submission_lgb['LogP']) / 2
    submission_cb.to_csv("submission_cb+lgb.csv", index=False)
    # Prepare additional data
    print("Prepare additional data")
    prepare_add_data(add_data_path, test_data)

    # Pretain D-MPNN model on add data
    print("Start pretraining D-MPNN")
    pretrain_dmpnn()

    # Fine-tune ensemble of D-MPNN models and make predictions on test set
    print("Start fine-tuning D-MPNN")
    dmpnn_predictions = fit_predict_dmpnn()
    submission_dmpnn = test_data.drop(columns=['SMILES'])
    submission_dmpnn['LogP'] = np.array(dmpnn_predictions).mean(axis=0)
    submission_dmpnn.to_csv("submission_dmpnn.csv", index=False)
    # Final predictions
    submission_dmpnn['LogP'] = (submission_dmpnn['LogP'] + submission_cb['LogP']) / 2
    submission_dmpnn.to_csv("submission_cb+lgb+dmpnn.csv", index=False)
    print("--- %s total seconds elapsed ---" % (time.time() - start_time))

    
if __name__ == "__main__":
    main()