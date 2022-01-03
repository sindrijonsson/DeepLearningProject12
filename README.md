# DeepLearningProject12
Created by: Sindri Jónsson (s202056) & Thorvaldur Ingi Ingimundarson (s202037)

The project was developed using in Google Drive using Colab notebooks. The main notebooks (.ipynb) pertaining to the work done in this project were then migrated to this GitHub repository.

__NOTE:__ The dataset provided and used in this project contains personal and sensitive data and can therefore not be shared within this repository. For this reason an end-to-end run through of training/testing the models can not be performed. Instead, the results from k-folds have been saved in the `K_fold` folder.

The main results of the project are summarized in the following notebooks:
* model_performance.ipynb
* diagnosis_correction.ipynb

Demo text 2

# Repository structure

```
\DEEPLEARNINGPROJECT12
│   DiseaseNet_with_kfold.ipynb
│   FeatureNet_with_kfold.ipynb
│   MTLNet_with_kfold.ipynb
│   README.md
│   SJ plotting.ipynb
│   TH MTLNet_DomainNet correction (NA).ipynb
│   TH MTL_DomainNet kfold (NA).ipynb
│   TH plotting (NA).ipynb
│   TH plotting.ipynb
│
├───Figures
│       Copy of Dataset_Distributions.eps
│       Dataset_Distributions.eps
│       DiseaseExample.eps
│       Kfold_CFM.eps
│       Kfold_CFM_NA.eps
│       Kfold_CFM_NA.png
│       Net Diag - DOMAIN.drawio.png
│       Net Diag - FINAL.drawio.png
│       Net Diag - MTL.drawio.png
│
├───Files
│   │   dermx_labels.csv
│   │   diseases_characteristics.csv
│   │
│   └───images
│           044401HB.jpeg
│
├───HelperFunctions
│       project_utils.py
│
└───K_fold
        Correction_FINAL_kfold_NA_0.json
        Correction_FINAL_kfold_NA_1.json
        Correction_FINAL_kfold_NA_2.json
        Correction_FINAL_kfold_NA_3.json
        Correction_FINAL_kfold_NA_4.json
        DiseaseNet_FINAL_kfold_NA_0.json
        DiseaseNet_FINAL_kfold_NA_1.json
        DiseaseNet_FINAL_kfold_NA_2.json
        DiseaseNet_FINAL_kfold_NA_3.json
        DiseaseNet_FINAL_kfold_NA_4.json
        Features_FINAL_kfold_NA_0.json
        Features_FINAL_kfold_NA_1.json
        Features_FINAL_kfold_NA_2.json
        Features_FINAL_kfold_NA_3.json
        Features_FINAL_kfold_NA_4.json
        MTLNet_FINAL_kfold_NA_0.json
        MTLNet_FINAL_kfold_NA_1.json
        MTLNet_FINAL_kfold_NA_2.json
        MTLNet_FINAL_kfold_NA_3.json
        MTLNet_FINAL_kfold_NA_4.json
        splits_FINAL_NA.csv
```
