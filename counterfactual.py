import pandas as pd
import torch
import dice_ml
from dataset.adult import Adult

#generate_CF takes a dictionary as an input which has the column name as its keys and the value as its corresponding values
#generate_CF returns a dataframe
def generate_CF(instance):
    ds_train = Adult('dataset', train=True).get_original_features()
    d = dice_ml.Data(dataframe=ds_train, continuous_features=['age', 'fnlwgt', 'educational-num', 'capital-gain','capital-loss', 'hours-per-week'], outcome_name='income')
    backend = 'PYT'
    ML_modelpath = "model_ori.pth"
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m)
    dice_exp = exp.generate_counterfactuals(instance, total_CFs=5, desired_class="opposite",
                                            proximity_weight=0.5, diversity_weight=1, categorical_penalty=0.1, 
                                            algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", 
                                            diversity_loss_type="dpp_style:inverse_dist", 
                                            feature_weights="inverse_mad", optimizer="pytorch:adam", 
                                            learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, 
                                            loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, 
                                            init_near_query_instance=True, tie_random=False, 
                                            stopping_threshold=0.5, posthoc_sparsity_param=0.1, 
                                            posthoc_sparsity_algorithm="binary")
    res = dice_exp.visualize_as_dataframe(show_only_changes=True)
    return res



def main():
    #exapmle
    res = generate_CF({'age': 0.682426, 'workclass': 'Private', 'fnlwgt': 0.852807, 'education': 'HS-grad', 'educational-num': -0.419331,'marital-status':'Married-civ-spouse', 'occupation': 'Machine-op-inspct', 'relationship': 'Husband', 'race':'White', 'gender':'Male','capital-gain':0.271595, 'capital-loss': -0.217125, 'hours-per-week':0.61152, 'native-country':'United-States'})
    print(res)

if __name__ == '__main__':
    main()
