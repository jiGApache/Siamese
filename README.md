# ECG-development
Development of models for ECG Recognition Library

## "ECG embedding classification" package
This package allows to reproduce the training and evaluation of the ML model. The model calculates the embeddings from normal and abnormal ECGs and produces their subsequent classification.
### Data Requirement
 - Required ECG frequency: 500 Hz
 - Required length: = 4s
### Usage examples
1. Embedding model training

    run ```python ECG_embedding_classification/TrainModel.py```
    This command trains embedding model and saves it's weights in ```Nets``` folder
2. Visualize ecg embeddings

    run ```python ECG_embedding_classification/VisualizeEmbeddings.py```
    
    Available flags:
    - ```-r``` or ```--reduction-method``` to reduce umbeddings dimension. Possible options: ```pca```, ```umap``` (used by default)
    - ```-d``` or ```--data-type``` to choose data to visualize. Possible options: ```train```, ```val```, ```test``` (used by default)
3. Test model with Few Shot approach

    run ```python ECG_embedding_classification/FewShotTestModel.py```

    Available flags:
    - ```-l``` or ```--labels``` to specify abnromal data label used in classification. Possible options: ```all``` (used by default), ```st```, ```mi```
    - ```-m``` or ```--metrics``` to choose metrics by which the model is evaluated. Possible options: ```accuracy``` (used by default), ```f1```