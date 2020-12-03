# Interactive website for Adult classifier explanations
How to use explanations with categorical variables: https://github.com/MaartenGr/InterpretableML/blob/master/InterpretableML.ipynb
I think you just sum up all explanations for the one-hot vector components

## Make the code run:
- Get Python 3.8 (e.g. with Anaconda)
- install PyTorch (only CPU): https://pytorch.org/
- Install custom additional packages: `pip install shap==0.36.0 dash plotly umap-learn matplotlib dice-ml catboost`

## TODO:
- data exploration (classification scatter plot, tabular insight into data values)
    - Raphael & Till
- displaying model quality and classification confidence
    - for different models (Hazem)!
- explanation insight for complete dataset and single data points (feature information)
    - Katharina & Matthias
- different explanation strategies?
    - Hanxiao
