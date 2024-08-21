import pandas as pd
import numpy as np
import os
import yaml
from TFE_localMeasures import GraphStartingPoints, GraphVisualizing
import random


#Expression datasets
for i in [17,23,26]:

    EXPPATH = f"Data/Dyngen/Data{i}/exp_data_processed.csv"
    if os.path.exists(EXPPATH):
        expdataset = pd.read_csv(EXPPATH,index_col=0)
    else:
        print(f"The file {EXPPATH} does not exist.")
        exit(-1)

    PATH = f"Data/Dyngen/Data{i}/reg_network_processed.csv"
    if os.path.exists(PATH):
        dataset = pd.read_csv(PATH,index_col=0)
        dataset = dataset.reset_index()
    else:
        print(f"The file {PATH} does not exist.")
        exit(-1)

    Interactions_Matrix = np.zeros((expdataset.columns.size,expdataset.columns.size))
    GraphVisualizing(dataset,f"Data/Dyngen/Data{i}/reg_network.png")

    for j in range(dataset.shape[0]):
        row = dataset.loc[j,'regulator']-1
        col = dataset.loc[j,'target']-1
        Interactions_Matrix[row,col] = 1

    
    #Generation of permuted networks for subtypes

    k = 10


    if not os.path.exists(f"Data/Dyngen/Data{i}/{k}_types"):
        os.makedirs(f"Data/Dyngen/Data{i}/{k}_types")

    for l in range(k):
        permuted_network = Interactions_Matrix.copy()
        #Probability of changes
        for m in range(expdataset.columns.size):
            if np.random.random() < 0.2:
                index = np.random.randint(0,expdataset.columns.size)
                permuted_network[m] = Interactions_Matrix[index]
                permuted_network[:,m] = Interactions_Matrix[:,index]
                permuted_network[index] = Interactions_Matrix[m]
                permuted_network[:,index] = Interactions_Matrix[:,m]
                
        permuted_network_dataframe = pd.DataFrame(columns=dataset.columns[1:])
        count = 0

        for n in range(expdataset.columns.size):
            for o in range(expdataset.columns.size-1):
                if permuted_network[n,o] == 1:
                    if n == o:
                        continue
                    else:
                        permuted_network_dataframe.loc[count] =  [n+1, o+1]
                    count += 1
        #Save the permuted network
        permuted_network_dataframe.to_csv(f"Data/Dyngen/Data{i}/{k}_types/permuted_network_{l}.csv")
        GraphVisualizing(permuted_network_dataframe,f"Data/Dyngen/Data{i}/{k}_types/permuted_network_{l}.png")

        file1 = f"Data/Dyngen/Data{i}/{k}_types/data{i}_{k}_net{l}.txt"
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(f"Data/Dyngen/Data{i}/{k}_types/data{i}_{k}_strength{l}.txt"):
            os.remove(f"Data/Dyngen/Data{i}/{k}_types/data{i}_{k}_strength{l}.txt")
        file2 = f"Data/Dyngen/Data{i}/{k}_types/data{i}_{k}_net{l}_strengths.txt"
        if os.path.exists(file2):
            os.remove(file2)

        with open(file1, mode='a', encoding='UTF-8') as output:
            with open(file2, mode='a', encoding='UTF-8') as output2:
                print("Gene\tRule",file=output)
                print("Gene1\tGene2\tStrength",file=output2)
                for p in range(1,expdataset.columns.size):
                    filtered_dataframe = permuted_network_dataframe.loc[permuted_network_dataframe['regulator'] == p]
                        
                    if filtered_dataframe.shape[0] == 0:
                        print(f"g{p}\tg{p}",file=output)
                        print(f"g{p}\t1",file=output2)
                    else:
                        print(f"g{p}\t",file=output, end="")
                        for q in range(filtered_dataframe.shape[0]):

                            if q != 0:
                                print(" and ",file=output, end="")

                            print(f"g{filtered_dataframe.iloc[q]["target"]}",file=output, end="")
                                
                            print(f"g{p}\t",file=output2, end="")
                            print(f"g{filtered_dataframe.iloc[q]["target"]}\t{np.random.randint(low=1,high=10)}",file=output2)
                            
                        print("\n",file=output, end="")
        #Initial conditions
        file3 = f"Data/Dyngen/Data{i}/{k}_types/data{i}_{k}_ics{l}.txt"
            
        if os.path.exists(file3):
            os.remove(file3)

        with open(file3, mode='a', encoding='UTF-8') as output3:
            print("Genes\tValues",file=output3)
            for r in GraphStartingPoints(permuted_network_dataframe):
                print(f"['g{r}']\t[1]",file=output3)

        # Define config file for boolode
        yamldata = {
            'global_settings': {
                'model_dir': f"Data/Dyngen/Data{i}/{k}_types",
                'output_dir': f"Data/Dyngen/Data{i}/{k}_types/boolode_output",
                'do_simulations': True,
                'do_post_processing': False,
                'modeltype': 'hill'
            },
            'jobs': [
                    {
                        'name': f"Data{i}_{k}_net{l}",
                        'model_definition': f"data{i}_{k}_net{l}.txt",
                        'simulation_time': 9,
                        'num_cells': 200,
                        'do_parallel': False,
                        'sample_cells': False
                    }
            ],
        }

        # Write to a YAML file
        with open(f'Code/BoolODE-master/config-files/Data{i}_{k}_net{l}.yaml', 'w') as file:
            yaml.dump(yamldata, file)