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

#Function to print scores for each method

def print_scores(method,norma,Interactions_Matrix, Interactions_Matrix_pred, length, sep = -1):
    for j in range(length):
        auroc_array = []
        pr_array = []
        print("Method : ",method[j])
        print("Normalized by : ",norma[j])

        #If working with separated datasets
        if sep >= 0:
            Pred = Interactions_Matrix_pred[j,0]
            GroundTruth = Interactions_Matrix[sep,0]
            precision, recall, thresholds = precision_recall_curve(GroundTruth,Pred)
            fpr, tpr, _ = roc_curve(GroundTruth,Pred)
            auroc_array.append(roc_auc_score(GroundTruth,Pred))
            pr_array.append(auc(recall,precision))
            print("Auroc for type", str(sep), " : ",auroc_array[0])
            print("AUPR for type ",str(sep), " : ",pr_array[0])
            print("-----------------------------------")

        #If working with the whole dataset
        elif method[j] != "Genie3" or method[j] != "globalMDA" or method[j] != "globalMDI":
            for a in range (0,10):
                Pred = np.mean(Interactions_Matrix_pred[j,a*200:(a+1)*200],axis=0)
                GroundTruth = Interactions_Matrix[a,0]
                precision, recall, thresholds = precision_recall_curve(GroundTruth,Pred)
                fpr, tpr, _ = roc_curve(GroundTruth,Pred)
                auroc_array.append(roc_auc_score(GroundTruth,Pred))
                pr_array.append(auc(recall,precision))
                print("Auroc for type", str(a), " : ",auroc_array[a])
                print( "AUPR for type ",str(a), " : ",pr_array[a])
        else:
            for a in range (0,10):
                Pred = Interactions_Matrix_pred[j,a*200]
                GroundTruth = Interactions_Matrix[a,0]
                precision, recall, thresholds = precision_recall_curve(GroundTruth,Pred)
                fpr, tpr, _ = roc_curve(GroundTruth,Pred)
                auroc_array.append(roc_auc_score(GroundTruth,Pred))
                pr_array.append(auc(recall,precision))
                print("Auroc for type", str(a), " : ",auroc_array[a])
                print( "AUPR for type ",str(a), " : ",pr_array[a])
        print("-----------------------------------")
    return

#Function to normalize and labelize the values of the local measures
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
sets = []

for type in range(0,10):
    PATH = f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/exp_data_processed.csv"
    if type == 0:
        train_set = pd.read_csv(PATH,index_col=0)
        train_set['Type'] = type
        sets.append(train_set)
    else:
        temp = pd.read_csv(PATH,index_col=0)
        temp['Type'] = type
        sets.append(temp)
        train_set = pd.concat([train_set,temp], axis=0)

#Generation of Train and test sets
print(f"Dataset{data_num} Splitting...  \n")
train_set_attributes_list = []
train_set_labels_list = []


#Keep cell ids and types for later
train_set.sort_values(by = ['cell_id'],inplace=True)
id_train_set = train_set['cell_id'].copy()
types_train_set = train_set['Type'].copy()
train_set.drop(columns=['cell_id'], inplace=True)
train_set.drop(columns=['Type'],inplace=True)

for i in range(train_set.columns.size):
    train_set_attributes_list.append(train_set.drop(train_set.columns[i], axis=1))
    train_set_labels_list.append(train_set[train_set.columns[i]].copy())

#for separated train_sets and labels
sep_train_set_attributes_list = []
sep_train_set_labels_list = []
counter = 0

for strain_set in sets:

    sep_train_set_attributes_list.append([])
    sep_train_set_labels_list.append([])
    #Keep cell ids and types for later
    strain_set.sort_values(by = ['cell_id'],inplace=True)
    id_train_set = strain_set['cell_id'].copy()
    types_train_set = strain_set['Type'].copy()
    strain_set.drop(columns=['cell_id'], inplace=True)
    strain_set.drop(columns=['Type'],inplace=True)

    for i in range(strain_set.columns.size):
        sep_train_set_attributes_list[counter].append(strain_set.drop(strain_set.columns[i], axis=1))
        sep_train_set_labels_list[counter].append(strain_set[strain_set.columns[i]].copy())
    counter+=1


"""
Load dataset and Interactions Matrix
"""
Interactions_Matrix = np.zeros((10,train_set.shape[0],train_set.columns.size,train_set.columns.size-1))
for type in range(0,10):
    RPATH = f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/reg_network_processed.csv"
    Rdataset = pd.read_csv(RPATH,index_col=0)
    row = Rdataset['regulator']-1
    col = Rdataset['target']-1
    for i in row.index:
        if col[i] < row[i] :
            Interactions_Matrix[type,:,row[i],col[i]] = 1
        else :
            Interactions_Matrix[type,:,row[i],col[i]-1] = 1

print("Shape of Interactions_Matrix : ",Interactions_Matrix.shape)
Interactions_Matrix = Interactions_Matrix.reshape(Interactions_Matrix.shape[0], Interactions_Matrix.shape[1], -1)
print("Shape of Interactions_Matrix : ",Interactions_Matrix.shape)



"""
Loads or train RF models
"""

print("Model training...  \n")
model_list = []

for i in range(train_set.columns.size):
    model_path = f"Data/Dyngen/Data{data_num}/10_types/model{data_num}_{i}.pkl"
    #If one model exists, it is assumed all models exists
    
    if os.path.exists(model_path):

        with open(model_path, 'rb') as file:
            model_list.append((joblib.load(file), train_set.columns[i]))

    else : 
        model = RandomForestRegressor(n_estimators=1000,max_features='sqrt',random_state=24,n_jobs=-1)
        model.fit(train_set_attributes_list[i].to_numpy(), train_set_labels_list[i].to_numpy())
        model_list.append((model, train_set.columns[i]))
        with open(f'Data/Dyngen/Data{data_num}/10_types/model{data_num}_{i}.pkl','wb') as file:
            joblib.dump(model,file)



#Local variable importance measures
print(f"Variable importance for dataset{data_num} measures computing... : \n")

#Construction of a 3D matrix containing the interactions between each pair of variables


    
Interactions_Matrix_pred = np.zeros((15,train_set.shape[0],train_set.columns.size,train_set.columns.size-1))
method_types = []
normal_types = []

for m in range(train_set.columns.size):
    #Constructs explainer for each
    model, target = model_list[m]
    
    localMAD_pertree_values = localMDA_parallel_pertree(model, train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("localMDA_parallel_pertree",method_types,normal_types,0,localMAD_pertree_values,Interactions_Matrix_pred,m)
    
    localMDA_tree_values = localMDAtree(model, train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("localMDA_tree",method_types,normal_types,5,localMDA_tree_values,Interactions_Matrix_pred,m)  
    
    shap_values = SHAPtree(model,train_set_attributes_list[m],train_set_attributes_list[m])
    normalize_and_labelize("SHAPtree",method_types,normal_types,10,abs(shap_values.values),Interactions_Matrix_pred,m)
    
    localMDA_values = localMDA_parallel(model, train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("localMDA_parallel",method_types,normal_types,15,localMDA_values,Interactions_Matrix_pred,m)
    
    _ , _, saabas_values = ti.predict(model, train_set_attributes_list[m].to_numpy())
    normalize_and_labelize("Saabas",method_types,normal_types,20,saabas_values,Interactions_Matrix_pred,m)
    
    localMDI_values = LocalMDI_cy.compute_mdi_local_ens(model, train_set_attributes_list[m].to_numpy().astype(np.float32))
    normalize_and_labelize("localMDI",method_types,normal_types,25,localMDI_values,Interactions_Matrix_pred,m)
    
    

Interactions_Matrix_pred = Interactions_Matrix_pred.reshape(Interactions_Matrix_pred.shape[0],Interactions_Matrix_pred.shape[1],
                                                            Interactions_Matrix_pred.shape[2]*Interactions_Matrix_pred.shape[3])
print_scores(method_types,normal_types,Interactions_Matrix, Interactions_Matrix_pred, 30)

#Global variable importance measures on separated datasets
counter  = 0
for element in sets:
    print("separated genie3 set", str(counter))
    VIM = GENIE3(element.to_numpy(),nthreads=16)
    VIM = VIM[~np.eye(VIM.shape[0],dtype=bool)].reshape(VIM.shape[0],-1)
    print(VIM.shape)
    method_types.append("Genie3")
    normal_types.append("None")
    Interactions_Matrix_pred[0,0,:] = VIM.reshape(-1)
    method_types.append("Genie3")
    normal_types.append("l1")
    Interactions_Matrix_pred[1,0,:] = normalize(VIM, norm='l1').reshape(-1) 
    method_types.append("Genie3")
    normal_types.append("l2")
    Interactions_Matrix_pred[2,0,:] = normalize(VIM, norm='l2').reshape(-1) 
    method_types.append("Genie3")
    normal_types.append("max")
    Interactions_Matrix_pred[3,0,:] = normalize(VIM, norm='max').reshape(-1) 
    method_types.append("Genie3")
    normal_types.append("minmax")
    Interactions_Matrix_pred[4,0,:]= minmax_scale(VIM, axis=1).reshape(-1)
    print_scores(method_types,normal_types,Interactions_Matrix, Interactions_Matrix_pred, 5, sep = counter)
    counter+=1
