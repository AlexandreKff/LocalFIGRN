import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_friedman1
from TFE_localMeasures import *
import sys
from GENIE3 import *
from treeinterpreter import treeinterpreter as ti
import LocalMDI_cy
from tqdm import tqdm
import joblib
import os
from scipy.stats import spearmanr
from FESP.attribution_models.attribution_scikit import *

# Function to print AUPR and AUROC scores
def print_scores(method,norma,Interactions_Matrix, Interactions_Matrix_pred, length):
    for j in range(length):
        auroc_array = []
        pr_array = []

        for i in range(Interactions_Matrix.shape[0]):
            GroundTruth = Interactions_Matrix[i]

            if method[j] != "Genie3" or method[j] != "globalMDA" or method[j] != "globalMDI":
                Pred = Interactions_Matrix_pred[j,i]
            else:
                Pred = Interactions_Matrix_pred[j,0]

            precision, recall, thresholds = precision_recall_curve(GroundTruth,Pred)
            fpr, tpr, _ = roc_curve(GroundTruth,Pred)
            auroc_array.append(roc_auc_score(GroundTruth,Pred))
            pr_array.append(auc(recall,precision))
        
        
        print("Method : ",method[j])
        print("Normalized by : ",norma[j])
        print("Auroc : ",np.mean(auroc_array))
        print( "AUPR : ",np.mean(pr_array))
        print(" Std aupr :", np.std(pr_array))
        print("-----------------------------------")
    return

# Function to normalize and labelize interaction matrices
def normalize_and_labelize(method,method_types,normal_types,starting_index,values,Interactions_Matrix_pred):
    method_types.append(method)
    normal_types.append("None")
    Interactions_Matrix_pred[starting_index,:,:] = values
    method_types.append(method)
    normal_types.append("l1")
    Interactions_Matrix_pred[starting_index+1,:,:] = normalize(values, norm='l1')
    method_types.append(method)
    normal_types.append("l2")
    Interactions_Matrix_pred[starting_index+2,:,:] = normalize(values, norm='l2') 
    method_types.append(method)
    normal_types.append("max")
    Interactions_Matrix_pred[starting_index+3,:,:] = normalize(values, norm='max') 
    method_types.append(method)
    normal_types.append("minmax")
    Interactions_Matrix_pred[starting_index+4,:,:]= minmax_scale(values, axis=1)

# Function to compute Spearman correlation
def compute_spearman(method,norma, Interactions_Matrix_pred, m1, m2):
    cor_array = np.zeros(Interactions_Matrix.shape[0])
        

    for i in range(Interactions_Matrix.shape[0]):
        cor_array[i] = spearmanr(Interactions_Matrix_pred[m1,i,:], Interactions_Matrix_pred[m2,i,:]).statistic
        if i ==1:
            print(cor_array[i])
    
    corr = np.mean(cor_array)
    
        
    print("Methods : ",method[m1], method[m2])
    print("Normalized by : ",norma[m1], norma[m2])
    print("Spearman correlation mean over samples : ", corr)
    print("-----------------------------------")
    return

# Function to compute feature importances with LES models
def _modelSexec(X, y, fesp_model, es_model, i):
    fesp_scores = fesp_model.fit(X[i].reshape(1,-1), y[i].reshape(1,-1))
    es_scores = es_model.fit(X[i].reshape(1,-1), y[i].reshape(1,-1))

    return (fesp_scores, es_scores,i)

"""
Loads dataset specified by data_num
"""
nf = 105

train_set_attributes, train_set_labels = make_friedman1(n_samples=1000, n_features=int(nf), noise=0.0, random_state=24)


"""
Loads or train RF model
"""                         

print("Model training...  \n")



model_path = f"Data/model_friedman1.pkl"
#If one model exists, it is assumed all models exists
    
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        modelf = joblib.load(file)

else : 
    modelf = RandomForestRegressor(n_estimators=1000, random_state=24)
    modelf.fit(train_set_attributes, train_set_labels)
    with open(f'Data/model_friedman1.pkl','wb') as file:
        joblib.dump(modelf,file)


#Construction of a 3D matrix containing the interactions between each pair of variables


    
Interactions_Matrix_pred = np.zeros((50, train_set_attributes.shape[0], train_set_attributes.shape[1]))
method_types = []
normal_types = []

print("Variable importance computing...")
#Constructs local variable importance measures
model = modelf

localMAD_pertree_values = localMDA_parallel_pertree(model, train_set_attributes)
normalize_and_labelize("localMDA_parallel_pertree",method_types,normal_types,0,localMAD_pertree_values,Interactions_Matrix_pred)

localMDA_tree_values = localMDAtree(model,train_set_attributes)
normalize_and_labelize("localMDA_tree",method_types,normal_types,5,localMDA_tree_values,Interactions_Matrix_pred)  

shap_values = SHAPtree(model,train_set_attributes,train_set_attributes)
normalize_and_labelize("SHAPtree",method_types,normal_types,10,abs(shap_values.values),Interactions_Matrix_pred)

localMDA_values = localMDA_parallel(model, train_set_attributes)
normalize_and_labelize("localMDA_parallel",method_types,normal_types,15,localMDA_values,Interactions_Matrix_pred)


fesp_model = FESPForScikit(model, retrain=False, baseline=0)
es_model = ESForScikit(model, retrain=False, baseline=0)

X = (train_set_attributes)
y = (train_set_labels)
fesp_scores = np.zeros((X.shape[0],train_set_attributes.shape[1]))
es_scores = np.zeros((X.shape[0],train_set_attributes.shape[1]))



v = Parallel(n_jobs=-1)(delayed(_modelSexec)(X.copy(), y.copy(), fesp_model, es_model, m) for m in range(X.shape[0]))  
for f,e,m in v:
    fesp_scores[m,:] = f
    es_scores[m,:] = e

normalize_and_labelize("FESP",method_types,normal_types,20,fesp_scores,Interactions_Matrix_pred)

normalize_and_labelize("ES",method_types,normal_types,25,es_scores,Interactions_Matrix_pred)

_ , _, saabas_values = ti.predict(model, train_set_attributes)
normalize_and_labelize("Saabas",method_types,normal_types,30,saabas_values,Interactions_Matrix_pred)


localMDI_values = LocalMDI_cy.compute_mdi_local_ens(model, train_set_attributes.astype(np.float32))
normalize_and_labelize("localMDI",method_types,normal_types,35,localMDI_values,Interactions_Matrix_pred)


globalMDA_values = permutation_importance(model,train_set_attributes, train_set_labels, n_jobs=-1,
                                             n_repeats=100, random_state=24)
normalize_and_labelize("globalMDA",method_types,normal_types,40,globalMDA_values.importances_mean.reshape(1,-1),Interactions_Matrix_pred)

#Genie3 approach
VIM = compute_feature_importances(model).reshape(1,-1)
method_types.append("Genie3")
normal_types.append("None")
Interactions_Matrix_pred[45,:] = VIM
method_types.append("Genie3")
normal_types.append("l1")
Interactions_Matrix_pred[46,:] = normalize(VIM, norm='l1') 
method_types.append("Genie3")
normal_types.append("l2")
Interactions_Matrix_pred[47,:] = normalize(VIM, norm='l2') 
method_types.append("Genie3")
normal_types.append("max")
Interactions_Matrix_pred[48,:] = normalize(VIM, norm='max') 
method_types.append("Genie3")
normal_types.append("minmax")
Interactions_Matrix_pred[49,:]= minmax_scale(VIM, axis=1)

Interactions_Matrix = np.zeros((train_set_attributes.shape[0], train_set_attributes.shape[1]))
Interactions_Matrix[:,:4] = 1
print_scores(method_types,normal_types,Interactions_Matrix, Interactions_Matrix_pred, 10)

#Spearman correlation
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 5)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 10)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 15)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 20)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 25)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 0, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 10)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 15)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 20)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 25)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 5, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 15)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 20)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 25)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 10, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 15, 20)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 15, 25)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 15, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 15, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 15, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 20, 25)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 20, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 20, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 20, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 25, 30)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 25, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 25, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 30, 35)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 30, 40)
compute_spearman(method_types,normal_types, Interactions_Matrix_pred, 35, 40)


