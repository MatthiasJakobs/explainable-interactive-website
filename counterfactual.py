import pandas as pd
import torch
import dice_ml
from model import FcNet
from dataset.adult import Adult, continous_columns

#generate_CF takes a dictionary as an input which has the column name as its keys and the value as its corresponding values
#generate_CF returns a dataframe
def generate_CF(instance):
    X, y = Adult('dataset', train=True).pandas()
    ds = pd.concat((X, y), axis=1)
    d = dice_ml.Data(dataframe=ds, continuous_features=continous_columns, outcome_name='income')
    backend = 'PYT'
    model = FcNet()
    # TODO: Load weights
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m)
    dice_exp = exp.generate_counterfactuals(instance, total_CFs=1, desired_class="opposite",
                                            proximity_weight=0.5, diversity_weight=1, categorical_penalty=0.1, 
                                            algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", 
                                            diversity_loss_type="dpp_style:inverse_dist", 
                                            feature_weights="inverse_mad", optimizer="pytorch:adam", 
                                            learning_rate=0.05, min_iter=500, max_iter=1000, project_iter=0, 
                                            loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, 
                                            init_near_query_instance=True, tie_random=False, 
                                            stopping_threshold=0.5, posthoc_sparsity_param=0.1, 
                                            posthoc_sparsity_algorithm="binary")
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    return dice_exp.final_cfs_df

def get_difference(ds1, ds2):
    changes = []
    for c_name in ds1.columns:
        value1 = ds1[c_name].iloc[0]
        value2 = ds2[c_name].iloc[0]
        if value1 != value2:
            changes.append([c_name, value1, value2])
    return changes
    

def main():
    #exapmle
    instance = {'age': 0.682426, 'workclass': 'Private', 'fnlwgt': 0.852807, 'education': 'HS-grad', 'educational-num': -0.419331,'marital-status':'Married-civ-spouse', 'occupation': 'Machine-op-inspct', 'relationship': 'Husband', 'race':'White', 'gender':'Male','capital-gain':0.271595, 'capital-loss': -0.217125, 'hours-per-week':0.61152, 'native-country':'United-States'}
    res = generate_CF(instance)
    print("-----------------")
    print(get_difference(res.drop('income', axis=1), pd.DataFrame(instance, index=[0])))

if __name__ == '__main__':
    main()
