# DeepLearningProject12
Created by: Sindri Jónsson (s202056) & Thorvaldur Ingi Ingimundarson (s202037)

The project was developed using in Google Drive using Colab notebooks. The main notebooks (.ipynb) pertaining to the work done in this project were then migrated to this GitHub repository.

__NOTE:__ The dataset provided and used in this project contains personal and sensitive data and can therefore not be shared within this repository. For this reason an end-to-end run through of training/testing the models can not be performed. Instead, the results from k-folds have been saved in the `K_fold` folder.

However, the models are defined, trained and tested in a 5-fold CV in the notebooks found under `Models_kfold`.  

The main results of the project are summarized in the following notebooks:
* model_performance_results.ipynb
* correction_results.ipynb

# Repository structure

```
\DEEPLEARNINGPROJECT12
│   correction_results.ipynb
│   model_performance_results.ipynb
│   README.md
│
├───Figures
│       Dataset_Distributions.eps
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
├───K_fold
│       DiseaseNet_FINAL_kfold_NA_0.json
│       DiseaseNet_FINAL_kfold_NA_1.json
│       DiseaseNet_FINAL_kfold_NA_2.json
│       DiseaseNet_FINAL_kfold_NA_3.json
│       DiseaseNet_FINAL_kfold_NA_4.json
│       Features_FINAL_kfold_NA_0.json
│       Features_FINAL_kfold_NA_1.json
│       Features_FINAL_kfold_NA_2.json
│       Features_FINAL_kfold_NA_3.json
│       Features_FINAL_kfold_NA_4.json
│       MTLNet_FINAL_kfold_NA_0.json
│       MTLNet_FINAL_kfold_NA_1.json
│       MTLNet_FINAL_kfold_NA_2.json
│       MTLNet_FINAL_kfold_NA_3.json
│       MTLNet_FINAL_kfold_NA_4.json
│       Preds_Correction_FINAL_kfold_NA_0.json
│       Preds_Correction_FINAL_kfold_NA_1.json
│       Preds_Correction_FINAL_kfold_NA_2.json
│       Preds_Correction_FINAL_kfold_NA_3.json
│       Preds_Correction_FINAL_kfold_NA_4.json
│       splits_FINAL_NA.csv
│       TrueFeatures_Correction_FINAL_kfold_NA_0.json
│       TrueFeatures_Correction_FINAL_kfold_NA_1.json
│       TrueFeatures_Correction_FINAL_kfold_NA_2.json
│       TrueFeatures_Correction_FINAL_kfold_NA_3.json
│       TrueFeatures_Correction_FINAL_kfold_NA_4.json
│
└───Models_kfold
        DiseaseNet_with_kfold.ipynb
        DomainNet_with_kfold.ipynb
        FeatureNet_with_kfold.ipynb
        MTLNet_DomainNet_Correction_kfold.ipynb
        MTLNet_with_kfold.ipynb
```

## Comments on running the notebooks:
* The notebooks were developed using Google Colab on Google Drive, therefore there are code snippets that try to mount to a secure drive.
* This will not be possible running locally, instead just comment out that block and make sure you are in the correct working directory.
* Make sure that all packages are installed.

