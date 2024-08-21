import pandas as pd
import numpy as np
import os
import re

for data_num in range(1,15):

    #Loads datasets un-processed
    EXPPATH = f"Data/Dyngen/Data{data_num}/exp_data.csv"
    REGPATH = f"Data/Dyngen/Data{data_num}/reg_network.csv"
    SCREGPATH = f"Data/Dyngen/Data{data_num}/reg_network_sc.csv"



    """
    Expression data for cells is composed of G columns, each representing a different gene
    """

    if os.path.exists(EXPPATH):
        Edataset = pd.read_csv(EXPPATH,index_col=0)
    else:
        print(f"The file {EXPPATH} does not exist.")
        exit(-1)

    ordinal_dict = {}

    column_order = 1
    for column in Edataset.columns:
        ordinal_dict[column] = column_order
        column_order += 1

    
    df = pd.DataFrame.from_dict(ordinal_dict, orient='index')
    df = df.transpose()
    df.to_csv(f"Data/Dyngen/Data{data_num}/ordinal_dict.csv", index=False)

    Edataset = Edataset.rename(columns=ordinal_dict)
    Edataset = Edataset.reset_index(drop=True)
    Edataset['cell_id'] = Edataset.index #Adding a column for cell_id

    """
    Average regulatory network is composed of two columns : regulator and its target

    Self-regulation interactions are neglected, because our method isnt properly designed to take them into account
    """

    if os.path.exists(REGPATH):
        Rdataset = pd.read_csv(REGPATH,index_col=0)
    else:
        print(f"The file {REGPATH} does not exist.")
        exit(-1)

    Rdataset.drop(columns = ['strength','effect'], inplace=True)
    Rdataset = Rdataset.reset_index(drop=True)
    selfRegulations = np.where(Rdataset['regulator'] == Rdataset['target'])[0]
    Rdataset.drop(index=selfRegulations,inplace = True)
    Rdataset = Rdataset.replace(ordinal_dict)


    """
    SC Reg networks are sorted according to their cell, and then represent three columns : regulator and its target, 
    and the effect of the link (-1 for downregulation, +1 otherwise).
    Strength are neglected for now.

    Self-regulation interactions are neglected, because our method isnt properly designed to take them into account
    """
    if os.path.exists(SCREGPATH):
        SCRdataset = pd.read_csv(SCREGPATH,index_col=0)
    else:
        print(f"The file {SCREGPATH} does not exist.")
        exit(-1)
        
    SCselfRegulations = np.where((SCRdataset['regulator'] == SCRdataset['target'])) #Self-Regulation edges
    SCRdataset.drop(index=SCselfRegulations[0],inplace = True)
    SCRdataset.drop(columns=['strength'], inplace=True)
    SCRdataset = SCRdataset.replace(ordinal_dict)
    SCRdataset = SCRdataset.reset_index(drop=True)
    
    for j in range(SCRdataset.shape[0]) :
        s = re.findall('\d+',SCRdataset['cell_id'][j])[0]
        SCRdataset.loc[j,'cell_id'] = int(s)

    SCRdataset.sort_values(by = ['cell_id'],inplace=True)
    SCRdataset = SCRdataset.reset_index(drop=True)

    Rdataset.to_csv(f"Data/Dyngen/Data{data_num}/reg_network_processed.csv")
    Edataset.to_csv(f"Data/Dyngen/Data{data_num}/exp_data_processed.csv")
    SCRdataset.to_csv(f"Data/Dyngen/Data{data_num}/reg_network_sc_processed.csv")


