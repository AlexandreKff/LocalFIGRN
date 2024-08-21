import pandas as pd
import numpy as np
import os
import re

for data_num in [17,23,26]:
    for type in range(0,10):

        #Loads datasets un-processed
        EXPPATH = f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/ExpressionData.csv"
        REGPATH = f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/refNetwork.csv"


        """
        Expression data for cells is composed of G columns, each representing a different gene
        """

        if os.path.exists(EXPPATH):
            Edataset = pd.read_csv(EXPPATH,index_col=0)
            Edataset = Edataset.transpose()
        else:
            print(f"The file {EXPPATH} does not exist.")
            exit(-1)

        #Keeps column order in memory
        ordinal_dict = {}

        column_order = 1
        for column in Edataset.columns:
            ordinal_dict[column] = column_order
            column_order += 1

        
        df = pd.DataFrame.from_dict(ordinal_dict, orient='index')
        df = df.transpose()
        df.to_csv(f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/ordinal_dict.csv", index=False)

        Edataset = Edataset.rename(columns=ordinal_dict)
        Edataset = Edataset.reset_index(drop=True)
        Edataset['cell_id'] = Edataset.index #Adding a column for cell_id

        """
        Average regulatory network is composed of two columns : regulator and its target

        Self-regulation interactions are neglected, because our method isnt properly designed to take them into account
        """

        if os.path.exists(REGPATH):
            Rdataset = pd.read_csv(REGPATH)
        else:
            print(f"The file {REGPATH} does not exist.")
            exit(-1)

        Rdataset.rename(columns={'Gene1': 'regulator', 'Gene2': 'target'}, inplace=True)
        Rdataset.drop(columns = ['Type'], inplace=True)
        Rdataset = Rdataset.reset_index(drop=True)
        selfRegulations = np.where(Rdataset['regulator'] == Rdataset['target'])[0]
        Rdataset.drop(index=selfRegulations,inplace = True)
        Rdataset = Rdataset.replace(ordinal_dict)



        Rdataset.to_csv(f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/reg_network_processed.csv")
        Edataset.to_csv(f"Data/Dyngen/Data{data_num}/10_types/boolode_output/Data{data_num}_10_net{type}/exp_data_processed.csv")
       