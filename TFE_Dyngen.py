import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.inspection import permutation_importance
from TFE_localMeasures import *
import sys
from GENIE3 import *
from treeinterpreter import treeinterpreter as ti
import LocalMDI_cy
from tqdm import tqdm
import joblib
import os

#Function to print scores of the different methods
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
        print("-----------------------------------")
    return

#Function to normalize and labelize the values of the different methods
def normalize_and_labelize(method,method_types,normal_types,starting_index,values,Interactions_Matrix_pred,m):
    method_types.append(method)
    normal_types.append("None")
    Interactions_Matrix_pred[starting_index,:,m,:] = values
    method_types.append(method)
    normal_types.append("l1")
    Interactions_Matrix_pred[starting_index+1,:,m,:] = normalize(values, norm='l1') 
    method_types.append(method)
    normal_types.append("l2")
    Interactions_Matrix_pred[starting_index+2,:,m,:] = normalize(values, norm='l2') 
    method_types.append(method)
    normal_types.append("max")
    Interactions_Matrix_pred[starting_index+3,:,m,:] = normalize(values, norm='max') 
    method_types.append(method)
    normal_types.append("minmax")
    Interactions_Matrix_pred[starting_index+4,:,m,:]= minmax_scale(values, axis=1)

"""
Loads dataset specified by data_num
"""
data_num = sys.argv[1]
PATH = f"Data/Dyngen/Data{data_num}/exp_data_processed.csv"

train_set = pd.read_csv(PATH,index_col=0)

#Generation of Train and test sets
print(f"Dataset{data_num} Splitting...  \n")
train_set_attributes_list = []
train_set_labels_list = []
small_train_set_attributes_list = []
small_train_set_labels_list = []
small_train_set = train_set.sample(frac=0.2, random_state=24)

#Keep cell ids for later
train_set.sort_values(by = ['cell_id'],inplace=True)
small_train_set.sort_values(by = ['cell_id'],inplace=True)
id_train_set = train_set['cell_id'].copy()
id_small_train_set = small_train_set['cell_id'].copy()
train_set.drop(columns=['cell_id'], inplace=True)
small_train_set.drop(columns=['cell_id'], inplace=True)

for i in range(train_set.columns.size):
    train_set_attributes_list.append(train_set.drop(train_set.columns[i], axis=1))
    train_set_labels_list.append(train_set[train_set.columns[i]].copy())
    small_train_set_attributes_list.append(small_train_set.drop(small_train_set.columns[i], axis=1))
    small_train_set_labels_list.append(small_train_set[small_train_set.columns[i]].copy())




"""
Load SC dataset and Interactions Matrix
"""
SCPATH = f"Data/Dyngen/Data{data_num}/reg_network_sc_processed.csv"
SCdataset = pd.read_csv(SCPATH,index_col=0)

Interactions_Matrix = np.zeros((train_set.shape[0],train_set.columns.size,train_set.columns.size-1))

for i in range(SCdataset.shape[0]):
    row = SCdataset.loc[i,'regulator']-1
    cell = SCdataset.loc[i,'cell_id']-1
    col = SCdataset.loc[i,'target']-1

    if col < row :
        Interactions_Matrix[cell,row,col] = 1
    else :
        Interactions_Matrix[cell,row,col-1] = 1
        
# Assuming id_test_set is a list of indices
Interactions_Matrix = Interactions_Matrix[id_small_train_set, :, :]
Interactions_Matrix = Interactions_Matrix.reshape(Interactions_Matrix.shape[0], -1)
print("Shape of Interactions_Matrix : ",Interactions_Matrix.shape)



"""
Loads or train RF models
"""

print("Model training...  \n")
model_list = []

for i in range(train_set.columns.size):
    model_path = f"Data/Dyngen/Data{data_num}/model{data_num}_{i}.pkl"
    #If one model exists, it is assumed all models exists

    if os.path.exists(model_path):

        with open(model_path, 'rb') as file:
            model_list.append((joblib.load(file), train_set.columns[i]))

    else : 
        model = RandomForestRegressor(n_estimators=1000,max_features='sqrt',random_state=24,n_jobs=-1)
        model.fit(train_set_attributes_list[i].to_numpy(), train_set_labels_list[i].to_numpy())
        model_list.append((model, train_set.columns[i]))
        with open(f'Data/Dyngen/Data{data_num}/model{data_num}_{i}.pkl','wb') as file:
            joblib.dump(model,file)



#Local variable importance measures
print(f"Variable importance for dataset{data_num} measures computing... : \n")

#Construction of a 3D matrix containing the interactions between each pair of variables

Interactions_Matrix_pred = np.zeros((5,small_train_set.shape[0],small_train_set.columns.size,small_train_set.columns.size-1))
method_types = []
normal_types = []

for m in range(train_set.columns.size):

    #Constructs explainer for each
    model, target = model_list[m]
    
    localMAD_pertree_values = localMDA_parallel_pertree(model, small_train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("localMDA_parallel_pertree",method_types,normal_types,0,localMAD_pertree_values,Interactions_Matrix_pred,m)
    
    localMDA_tree_values = localMDAtree(model, small_train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("localMDA_tree",method_types,normal_types,5,localMDA_tree_values,Interactions_Matrix_pred,m)  
    
    shap_values = SHAPtree(model,train_set_attributes_list[m],small_train_set_attributes_list[m])
    normalize_and_labelize("SHAPtree",method_types,normal_types,10,abs(shap_values.values),Interactions_Matrix_pred,m)
    
    localMDA_values = localMDA_parallel(model, small_train_set_attributes_list[m].to_numpy())
    print("Local MDA values : ", localMDA_values)
    normalize_and_labelize("localMDA_parallel",method_types,normal_types,15,localMDA_values,Interactions_Matrix_pred,m)

    _ , _, saabas_values = ti.predict(model, small_train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("Saabas",method_types,normal_types,20,saabas_values,Interactions_Matrix_pred,m)

    localMDI_values = LocalMDI_cy.compute_mdi_local_ens(model, small_train_set_attributes_list[m].to_numpy().astype(np.float32))
    normalize_and_labelize("localMDI",method_types,normal_types,25,localMDI_values,Interactions_Matrix_pred,m)

    globalMDA_values = permutation_importance(model,small_train_set_attributes_list[m].to_numpy(), small_train_set[target].to_numpy(),
                                             n_repeats=100, random_state=24)
    normalize_and_labelize("globalMDA",method_types,normal_types,30,globalMDA_values.importances_mean.reshape(1,-1),Interactions_Matrix_pred,m)
    
#Genie3 approach
VIM = GENIE3(small_train_set.to_numpy())
VIM = VIM[~np.eye(VIM.shape[0],dtype=bool)].reshape(VIM.shape[0],-1) #Remove diagonal values
method_types.append("Genie3")
normal_types.append("None")
Interactions_Matrix_pred[35,:] = VIM
method_types.append("Genie3")
normal_types.append("l1")
Interactions_Matrix_pred[36,:] = normalize(VIM, norm='l1') 
method_types.append("Genie3")
normal_types.append("l2")
Interactions_Matrix_pred[37,:] = normalize(VIM, norm='l2') 
method_types.append("Genie3")
normal_types.append("max")
Interactions_Matrix_pred[38,:] = normalize(VIM, norm='max') 
method_types.append("Genie3")
normal_types.append("minmax")
Interactions_Matrix_pred[39,:]= minmax_scale(VIM, axis=1)

Interactions_Matrix_pred = Interactions_Matrix_pred.reshape(Interactions_Matrix_pred.shape[0],Interactions_Matrix_pred.shape[1],
                                                            Interactions_Matrix_pred.shape[2]*Interactions_Matrix_pred.shape[3])
print_scores(method_types,normal_types,Interactions_Matrix, Interactions_Matrix_pred, 40)
