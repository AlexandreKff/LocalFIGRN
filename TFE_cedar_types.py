import pandas as pd
import os
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

#Prints the top 100 interactions for each method
def print_scores(method_types,normal_types, Interactions_Matrix_pred, nb_methods, nrows, mapping, types_train_set, all_types = True):
    for j in range(nb_methods):
        auroc_array = []
        pr_array = []
        print("Method : ",method_types[j])
        print("Normalized by : ",normal_types[j])

        if all_types == False:
            Pred = Interactions_Matrix_pred[j,0]
                
            # Reshape Pred if necessary
            if Pred.shape == (1, 1482):
                Pred = Pred.reshape(-1)
            
            # Get the indexes of the top 100 interactions
            top_100_indexes = np.argsort(Pred)[-100:]
            
            for i in top_100_indexes:
                col = i % (nrows-1)
                row = (i // (nrows-1))
                if col >= row:
                    col+=1
                print("Gene pair : ", mapping[col], " -> ", mapping[row])

        elif method_types[j] != "Genie3" or method_types[j] != "globalMDA" or method_types[j] != "globalMDI":
            for a in range (0,27):
                ind_type_a = np.where(types_train_set == a)
                Pred = np.mean(Interactions_Matrix_pred[j,ind_type_a[0]],axis=0)
                print("--------------")
                print("Type ", a)
                print("--------------")
                # Reshape Pred if necessary
                if Pred.shape == (1, 1482):
                    Pred = Pred.reshape(-1)

                # Get the indexes of the top 100 interactions
                top_100_indexes = np.argsort(Pred)[-100:]
                
                for i in top_100_indexes:
                    col = i % (nrows-1)
                    row = (i // (nrows-1))
                    if col >= row:
                        col+=1
                    print("Gene pair : ", mapping[col], " -> ", mapping[row])

        else:
            for a in range (0,27):
                ind_type_a = np.where(types_train_set == a)
                Pred = Interactions_Matrix_pred[j,ind_type_a[0,0]]

                    
                # Reshape Pred if necessary
                if Pred.shape == (1, 1482):
                    Pred = Pred.reshape(-1)
                print("--------------")
                print("Type ", a)
                print("--------------")
                # Get the indexes of the top 100 interactions
                top_100_indexes = np.argsort(Pred)[-100:]
                
                for i in top_100_indexes:
                    col = i % (nrows-1)
                    row = (i // (nrows-1))
                    if col >= row:
                        col+=1
                    print("Gene pair : ", mapping[col], " -> ", mapping[row])

        print("-----------------------------------")
    return

#Function to normalize and labelize interaction matrices
def normalize_and_labelize(method,method_types,normal_types,starting_index,values,Interactions_Matrix_pred,m):
    method_types.append(method)
    normal_types.append("None")
    Interactions_Matrix_pred[starting_index,:,m,:] = values


# Define the list of datasets (#types)
folder_path = "Data/DataCEDAR/expression_data/"
all_files_and_dirs = os.listdir(folder_path)
files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]

# Define the path to the text file
gi_path = "Data/DataCEDAR/list_genes_colocalized_with_IBD_risk_loci.txt"
lines_array = []

# List of genes of interest
with open(gi_path, 'r') as file:
    lines_array = [line.strip() for line in file]

merged_bed = pd.DataFrame()

for i, file in enumerate(files):
    path = os.path.join(folder_path, file)
    bed = pd.read_csv(path, delimiter='\t')
    bed = bed.transpose()
    bed.columns = bed.iloc[3]
    bed = bed[6:]
    bed = bed[[col for col in bed.columns if col in lines_array]]

    # Add a column indicating the dataset type
    bed['type'] = i

    if merged_bed.empty:
            merged_bed = bed
    else:
        # Find common columns
        common_columns = merged_bed.columns.intersection(bed.columns)
        merged_bed = merged_bed[common_columns]
        bed = bed[common_columns]
        # Concatenate the current dataset to the merged dataframe
        merged_bed = pd.concat([merged_bed, bed], axis=0)


# Save the merged dataframe to a csv file
merged_bed.to_csv("Data/DataCEDAR/merged_bed.csv", index=False)
# Save the index to columns mapping and save the types

index_to_columns_mapping = {i: col for i, col in enumerate(merged_bed.columns)}
merged_bed.columns = range(merged_bed.shape[1])


train_set = merged_bed.copy()
print("Train set shape : ", train_set.shape)

#for separated train_sets genie3 method
sep_train_set_attributes_list = []
sep_train_set_labels_list = []
sets = []

for counter in range(0,len(files)):
    sep_train_set_attributes_list.append([])
    sep_train_set_labels_list.append([])
    strain_set = train_set[train_set.iloc[:, -1] == counter].copy()
    types_train_set = strain_set.iloc[:, -1].copy()
    strain_set.drop(columns=[strain_set.columns[-1]],inplace=True)
    sets.append(strain_set)
    for i in range(strain_set.columns.size):
        sep_train_set_attributes_list[counter].append(strain_set.drop(strain_set.columns[i], axis=1))
        sep_train_set_labels_list[counter].append(strain_set[strain_set.columns[i]].copy())
types_train_set = train_set.iloc[:, -1].copy()
train_set.drop(columns=[train_set.columns[-1]],inplace=True)
# for the whole train_set for localMDI
train_set_attributes_list = []
train_set_labels_list = []
for i in range(train_set.columns.size):
    train_set_attributes_list.append(train_set.drop(train_set.columns[i], axis=1))
    train_set_labels_list.append(train_set[train_set.columns[i]].copy())

"""
Loads or train RF models
"""

print("Model training...  \n")
model_list = []

for i in range(train_set.columns.size):
    model_path = f"Data/DataCEDAR/model_{i}.pkl"
    #If one model exists, it is assumed all models exists
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model_list.append((joblib.load(file), train_set.columns[i]))

    else : 
        model = RandomForestRegressor(n_estimators=1000,max_features='sqrt',random_state=24,n_jobs=-1)
        model.fit(train_set_attributes_list[i].to_numpy(), train_set_labels_list[i].to_numpy())
        model_list.append((model, train_set.columns[i]))
        with open(f"Data/DataCEDAR/model_{i}.pkl",'wb') as file:
            joblib.dump(model,file)

#Local variable importance measures
print(f"Variable importance for dataset measures computing... : \n")

#Construction of a 3D matrix containing the interactions between each pair of variables


    
Interactions_Matrix_pred = np.zeros((1,train_set.shape[0],train_set.columns.size,train_set.columns.size-1))
method_types = []
normal_types = []

#localMDI method on whole train set
for m in range(train_set.columns.size):
    
    model, target = model_list[m]

    localMDI_values = LocalMDI_cy.compute_mdi_local_ens(model, train_set_attributes_list[m].to_numpy().astype(np.float32))
    normalize_and_labelize("localMDI",method_types,normal_types,0,localMDI_values,Interactions_Matrix_pred,m)

Interactions_Matrix_pred = Interactions_Matrix_pred.reshape(Interactions_Matrix_pred.shape[0],Interactions_Matrix_pred.shape[1],
                                                            Interactions_Matrix_pred.shape[2]*Interactions_Matrix_pred.shape[3])
print_scores(method_types,normal_types, Interactions_Matrix_pred, 1, train_set.columns.size, index_to_columns_mapping, types_train_set)

counter  = 0

#Genie3 method on separated train sets
for element in sets:
    print("separated genie3 set", str(counter))
    VIM = GENIE3(element.to_numpy(),nthreads=16)
    VIM = VIM[~np.eye(VIM.shape[0],dtype=bool)].reshape(VIM.shape[0],-1)
    print(VIM.shape)
    method_types.append("Genie3")
    normal_types.append("None")
    Interactions_Matrix_pred[0,0,:] = VIM.reshape(-1)

    print_scores(method_types,normal_types, Interactions_Matrix_pred, 1, train_set.columns.size, index_to_columns_mapping, all_types = False)
    counter+=1
