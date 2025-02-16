# TIACBM
This is the preliminary code base for TIACBM, which will be updated later. 

The main TIACBM experiments with their respective best parameters and curricula, can be found in each folder. 

- Prerequisites need to be installed with **pip install -r requirements.txt**
- Experiments using Reuters-21578 require adding the data set manually from: https://www.kaggle.com/nltkdata/reuters (version used by CB-NTR baseline), in a "data" folder, next to the main reuters experiment. The paths can also be changed inside the main files at the beginning.
- Experiments using SST2 require adding the SentiWordNet 3.0 lexicon inside the respective folder, found at: https://github.com/aesuli/SentiWordNet.
- Experiments using PAN19 require changing the "DATA_PATH" at the start of the main file, which links to the path of Problem 00001/00005 downloaded from: https://zenodo.org/records/3530313.
- For running the baselines, mask_probs needs to be adjusted to zeros, thus skipping the masking step.

Beyond those steps, all TIACBM experiments can be run with **python path/to/experiment.py**.

The source files for the Cart-(Stra)-CL++ baselines can be found at TIACBM/cartography_train. Depending on the data set and model, the user needs to gather the sorted json first. Afterwards, at TIACBM/cartography_train/cart_(stra)_cl++/ the user needs to adjust the path to the respective sorted json, and change the learning rates as written in the paper, in case a joint file is used (PAN19). 
