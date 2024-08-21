# Import necessary modules and functions
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
from scipy.stats import spearmanr

# Function to print correlation between predicted interaction matrices
def print_corr(method_types, normal_types, Interactions_Matrix_pred, nb_methods, types_train_set, all_types=True):
    # Loop over each method
    for j in range(nb_methods):
        # Print the method type and normalization type
        print("Method : ", method_types[j])
        print("Normalized by : ", normal_types[j])

        # If not all types are considered
        if all_types == False:
            # Initialize Pred array to store predictions
            Pred = np.zeros((3, Interactions_Matrix_pred.shape[3]))
            # Store predictions for the first type
            Pred[0] = Interactions_Matrix_pred[j, 0, 0]
            # Store mean predictions for the second and third types
            Pred[1] = np.mean([Interactions_Matrix_pred[j, 1, 0], Interactions_Matrix_pred[j, 2, 0]], axis=0)
            # Compute Spearman correlation between the two types
            corr = spearmanr(Pred[0], Pred[1]).statistic
            # Print the correlation
            print("Correlation between the two types : ", corr)
        else:
            # Initialize Pred array to store predictions
            Pred = np.zeros((3, Interactions_Matrix_pred.shape[2]))
            # Loop over each type
            for a in range(0, 3):
                # Find indices of the current type in the training set
                ind_type_a = np.where(types_train_set == a)
                # Store mean predictions for the current type
                Pred[a] = np.mean(Interactions_Matrix_pred[j, ind_type_a[0]], axis=0)
            # Compute mean predictions for the second and third types
            Pred[1] = np.mean(Pred[1:], axis=0)
            # Compute Spearman correlation between the first type and the mean of the second and third types
            corr = spearmanr(Pred[0], Pred[1]).statistic
            # Print the correlation
            print("Correlation between the two types : ", corr)
        # Print separator
        print("-----------------------------------")
    return

# Function to normalize and labelize interaction matrices
def normalize_and_labelize(method, method_types, normal_types, starting_index, values, Interactions_Matrix_pred, m):
    # Append the method type to the method_types list
    method_types.append(method)
    # Append the normalization type to the normal_types list
    normal_types.append("None")
    # Normalize the values
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    # Update the interaction matrix with the normalized values
    Interactions_Matrix_pred[starting_index, :, m, :] = values

# Define the folder path containing the expression data files
folder_path = "Data/DataCEDAR/expression_data/"

# List all files and directories in the specified folder
all_files_and_dirs = os.listdir(folder_path)

# Filter out only the files from the list of all files and directories
files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]

# Define the path to the text file containing the list of genes of interest
gi_path = "Data/DataCEDAR/list_genes_colocalized_with_IBD_risk_loci.txt"

# Initialize an empty list to store the lines from the text file
lines_array = []

# Read the list of genes of interest from the text file
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
# Define the path to the text file of genetic markers
ge_path = "Data/DataCEDAR/genotypes_at_top_SNP_of_IBD_risk_loci.txt"
ge_markers = pd.read_csv(ge_path, delimiter='\t')
ge_markers = ge_markers.transpose()
types_train_set = []
for gmk in range(ge_markers.shape[1]):
    #Add the genetic markers to the train_set
    column = ge_markers.columns[gmk]
    for patient in ge_markers.index.tolist():
        train_set.loc[patient,column] = int(ge_markers.loc[patient,column])
    
    types_train_set.append(train_set.iloc[:, -1].copy())
    train_set.drop(columns=[train_set.columns[-1]],inplace=True)

#for separated train_sets
sep_train_set_attributes_list = []
sep_train_set_labels_list = []
sets = []

# for the whole train_set
train_set_attributes_list = []
train_set_labels_list = []
for i in range(train_set.columns.size): ####
    train_set_attributes_list.append(train_set.drop(train_set.columns[i], axis=1))
    train_set_labels_list.append(train_set[train_set.columns[i]].copy())


#Loads or train RF models


print("Model training...  \n")
model_list = []

for i in range(train_set.columns.size): ####
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
    
Interactions_Matrix_pred = np.zeros((1,train_set.shape[0],train_set.columns.size,train_set.columns.size-1))####
method_types = []
normal_types = []

for m in range(train_set.columns.size): ####
    #Constructs explainer for each
    model, target = model_list[m]
    
    localMDI_values = LocalMDI_cy.compute_mdi_local_ens(model, train_set_attributes_list[m].to_numpy().astype(np.float32))
    normalize_and_labelize("localMDI",method_types,normal_types,0,localMDI_values,Interactions_Matrix_pred,m)
    



Interactions_Matrix_pred = Interactions_Matrix_pred.reshape(Interactions_Matrix_pred.shape[0],Interactions_Matrix_pred.shape[1],
                                                            Interactions_Matrix_pred.shape[2]*Interactions_Matrix_pred.shape[3])
for tp in types_train_set:
    print_corr(method_types,normal_types, Interactions_Matrix_pred, 1, tp)


