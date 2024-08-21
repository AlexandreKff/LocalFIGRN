import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import networkx as nx
from tqdm import tqdm
import time
import random
from joblib import Parallel, delayed


# Function to calculate the local measures of importance for a given model and dataset using SHAP values as in https://github.com/shap/shap
def SHAPtree(model, train_data, test_data):
    SHAPexplainer = shap.Explainer(model, train_data)
    return SHAPexplainer(test_data, check_additivity=False)

# Function to calculate the local measures of importance for a given model and dataset using localMDA forest perturbation
def _localMDA_permutation(data_perm, n_iter, model, m, Ypred, random_seed = 24): #Erreur de la moyenne
    np.random.seed(random_seed)
    Ypred_perm = np.zeros((n_iter,data_perm.shape[0]))
    for i in range(n_iter):
        np.random.shuffle(data_perm[:,m])
        Ypred_perm[i] = model.predict(data_perm)

    return ((((np.mean(Ypred_perm,axis=0)) - Ypred)**2),m)

# Function to calculate the local measures of importance for a given model and dataset using localMDA tree perturbation
def _localMDA_permutation_pertree(data_perm, model, m, Ypred, random_seed = 24): #Moyenne des erreurs
    np.random.seed(random_seed)
    importances = []
    means = []
    temp = []
    for tree in model.estimators_:
        np.random.shuffle(data_perm[:,m])
        Ypred_perm = tree.predict(data_perm)
        importances.append((Ypred_perm - Ypred)**2)

    for i in range(data_perm.shape[0]):
        temp.append([])
        for j in range(len(importances)):
            temp[i].append(importances[j][i])
        means.append(np.mean(temp[i]))

    return (means,m) 

# Function to calculate the local measures of importance for a given model and dataset using localMDA tree structure. It is based on https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html algorithm
def _localMDA_tree(data, model, m, Ypred): #Moyenne des erreurs de chaque arbre
    feature_imp = np.zeros((data.shape[0]))
    for sample in range(data.shape[0]):
        sample_pred = []
        for tree in model.estimators_:
            leaves = []
            tested_feature = m
            model_tree = tree.tree_
            children_left = model_tree.children_left
            children_right = model_tree.children_right
            n_nodes_tree = model_tree.n_node_samples
            feature = model_tree.feature
            threshold = model_tree.threshold
            values = model_tree.value
            stack = [(0,0,1)]

            while len(stack) > 0:
                node_id, depth, weighted_value = stack.pop()

                if(children_right[node_id] == children_left[node_id]):
                    leaves.append(weighted_value*values[node_id])
                elif(feature[node_id] == tested_feature):
                    stack.append((children_left[node_id],depth+1,weighted_value*(n_nodes_tree[children_left[node_id]]/n_nodes_tree[node_id])))
                    stack.append((children_right[node_id],depth+1,weighted_value*(n_nodes_tree[children_right[node_id]]/n_nodes_tree[node_id])))

                elif(data[sample][feature[node_id]] <= threshold[node_id]):
                    stack.append((children_left[node_id],depth+1,weighted_value))
                else:
                    stack.append((children_right[node_id],depth+1,weighted_value))
            sample_pred.append((np.sum(leaves)- Ypred[sample])**2)
        feature_imp[sample] += np.mean(sample_pred)
    return (feature_imp,m)

# Function to calculate the local measures of importance for a given model and dataset using localMDA forest perturbations
def localMDA_parallel(model, data, n_iter = 100, random_seed = 24, n_jobs = 8):
    feature_imp = np.zeros((data.shape[0],data.shape[1]))

    Ypred = model.predict(data)

    v = Parallel(n_jobs=8)(delayed(_localMDA_permutation)(data.copy(), n_iter, model, m, Ypred, random_seed) for m in range(data.shape[1]))
    for values,m in v:
        feature_imp[:,m] = values
    return feature_imp

# Function to calculate the local measures of importance for a given model and dataset using localMDA tree perturbations
def localMDA_parallel_pertree(model, data, random_seed = 24, n_jobs = 16):
    feature_imp = np.zeros((data.shape[0],data.shape[1]))
    v = []
    Ypred = model.predict(data)


    v = Parallel(n_jobs=n_jobs)(delayed(_localMDA_permutation_pertree)(data.copy(), model, m, Ypred, random_seed) for m in range(data.shape[1]))  
    for values,m in v:
        feature_imp[:,m] = values

    return feature_imp

# Function to calculate the local measures of importance for a given model and dataset using localMDA tree structure
def localMDAtree(model, data, n_jobs = 16):
    feature_imp = np.zeros((data.shape[0],data.shape[1]))

    Ypred = model.predict(data)
    

    v = Parallel(n_jobs=n_jobs)(delayed(_localMDA_tree)(data.copy(), model, m, Ypred) for m in range(data.shape[1]))
    for values,m in v:
        feature_imp[:,m] = values
    return feature_imp

#function to visualize the graph of the datasets and/or starting points. Functions were useful to generate networks permutations and initial conditions
def GraphVisualizing(data,filename):
    # Create a new directed graph
    G = nx.from_pandas_edgelist(data, 'regulator', 'target', create_using=nx.DiGraph())

    # Draw the entire graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='r', node_size=500, font_size=10)

    # Save the figure
    plt.savefig(filename)
    plt.close()

def GraphStartingPoints(data):
    random.seed(24)
    # Create a directed graph from the dataframe
    G = nx.from_pandas_edgelist(data, 'regulator', 'target', create_using=nx.DiGraph())

    # Find weakly connected components
    wcc = list(nx.weakly_connected_components(G))

    # Create an empty list to store the starting points
    starting_points_list = []

    # For each weakly connected component
    for component in wcc:
        # Create a subgraph
        subgraph = G.subgraph(component)
        # Find nodes with in-degree 0
        starting_points = [node for node, degree in subgraph.in_degree() if degree == 0]
        # If there are no such nodes, then the component is a cycle
        if not starting_points:
            # Choose a random node from the cycle
            starting_points = [random.choice(list(component))]
        # Append the starting points to the list
        starting_points_list.extend(starting_points)

    # Return the list of starting points
    return starting_points_list