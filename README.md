# Sibur_element_119_2025
Final solution of Sibur Element 119 challenge (prediction of lipophilicity value (LogP) of small organic molecules by SMILES).

# Description
Directory ```data``` contains train and test csv-files. In addition, the directory contains files with [ALOGPS 2.1](https://vcclab.org/lab/alogps/start.html) descriptors for train and test subsets.
Additional data used to pretain D-MPNN model also is located in ```data``` directory (```alogps_3_01_training_.csv```). This data can be downloaded from [OCHEM database](https://ochem.eu/home/show.do).
# Create environment
```git clone https://github.com/zxcghoul3228/Sibur_element_119_2025.git```
```
cd Sibur_element_119_2025
conda create -n Sibur python==3.12.9
conda activate Sibur
pip install -r requirements.txt
```
# Run training script
```
python main.py
```
