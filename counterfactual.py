import pandas as pd
import torch
import dice_ml
from model import FcNet
from dataset.adult import Adult, continous_columns

#generate_CF takes a Dataframe as an input and returns a Dataframe with a series of CFs
def generate_CF(instance):
    X, y = Adult('dataset', train=True).pandas()
    ds = pd.concat((X, y), axis=1)
    d = dice_ml.Data(dataframe=ds, continuous_features=continous_columns, outcome_name='income')
    backend = 'PYT'
    model = FcNet()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m)
    instance = pd.DataFrame.to_dict(instance,orient ='record')
    dice_exp = exp.generate_counterfactuals(instance[0], total_CFs=1, desired_class="opposite",
                                            proximity_weight=0.5, diversity_weight=1, categorical_penalty=0.1, 
                                            algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", 
                                            diversity_loss_type="dpp_style:inverse_dist", 
                                            feature_weights="inverse_mad", optimizer="pytorch:adam", 
                                            learning_rate=0.05, min_iter=500, max_iter=1000, project_iter=0, 
                                            loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, 
                                            init_near_query_instance=True, tie_random=False, 
                                            stopping_threshold=0.5, posthoc_sparsity_param=0.1, 
                                            posthoc_sparsity_algorithm="binary")
    res = dice_exp.final_cfs_df
    return res


if __name__ == "__main__":
    #Example
    ori_dict = {'age': 0.682426, 'workclass': 'Private', 'fnlwgt': 0.852807, 'education': 'HS-grad', 'educational-num': -0.419331,'marital-status':'Married-civ-spouse', 'occupation': 'Machine-op-inspct', 'relationship': 'Husband', 'race':'White', 'gender':'Male','capital-gain':0.271595, 'capital-loss': -0.217125, 'hours-per-week':0.61152, 'native-country':'United-States'}
    instance = pd.DataFrame(ori_dict,index=[0])
    res = generate_CF(instance)
    print(res)
