# Local machine learning-based feature importances for gene regulatory network inference

This repository accounts for the master thesis "Local machine learning-based feature importances for gene regulatory network
inference" by Alexandre Kerff, 2023-2024, Uliège.

--------------





## Description

Understanding how a cell (or organism) reacts to a change in the environment or disturbance requires an understanding of the intricate processes controlling gene expression and, therefore, protein synthesis. A common representation of these mechanisms is the gene regulatory network, that aims at defining the regulation links between genes as a set of interactions. Inferring those gene regulatory networks from expression data has been a widely studied field at the level of bulk expression data. However, recent breakthroughs in sequencing technologies enables measurements at the resolution of a single cell. Such data allows the development of research towards the analysis of gene regulatory networks for a single specific cell or for a distinct cell type, rather than global interactions. This thesis has the objective to perform these analyses.

---------------
## Foreword

This repository contains all the codes that were necessary to find the results given in the master thesis. Be however aware that execution times for gene network inference methods might be important and that 90% of the codes needed the usage of CECI clusters to be run in reasonable amount of time.

## Project structure
The project requires a Dataset file (available at https://zenodo.org/records/13352287?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE2YzhlYzdmLTEzZjAtNDQ4Zi05NTRlLTA0NGU5MGEyZWQwNCIsImRhdGEiOnt9LCJyYW5kb20iOiJhYmI0Zjg2ZjAxMzg5ZmZjMGVhNjVmMmI5YWU3NGVkNyJ9.oG7B5lpZcrO-7w5tk9PKbAKOUD0ydnQ7CX558j7LoZhjw_SAYZLAL5B-gctSN-O7kRl6bY6QS8UWyJJB0-qNyw) containing all the datasets used in this thesis. It must be unzipped in the original Data file in the current folder. Together is provided the saved model for friedman dataset and a set of commands that were used to generate the datasets with dyngen. The Dyngen file contains the 14 datasets used in dyngen (ranging from #15 to #28) with processed files (with corresponding given python codes). In dataset 17, 23 and 26, it also contains the 10_types folder that contains the data concerning the cell-type inference problem. Not all ML saved models are published by lack of space. Finally, CEDAR dataset and concerned files is also available in the corresponding folder.

GENIE3 contains the code of the GENIE3 package defined in https://github.com/vahuynh/GENIE3.
FESP contains the code of the FESP package (with slight compatibility modifications) defined in https://github.com/ccdv-ai/fesp_es/tree/main.$

The codes are the following :
- TFE_friedmann.py : Contains the code needed to run the analysis of friedman dataset
- TFE_localMeasures.py : Contains the local feature importance methods
- TFE_Dyngen.py : Contains the dyngen dataset analysis methodology
- TFE_DyngenDataProcessing.py, TFE_permutationData(Post)Processing.py : contains the manipulation for processing all the data (already processed here)
- TFE_permutation.py : Contains the code needed to run the cell-type inference problem
- TFE_cedar_types and TFE_cedar_gemarkers : contains the code needed to run the cedar dataset analysis


## Installation and execution

A yaml file containing the exported conda environment called environment is available to install the same work environment. 
Once the environment is set, the steps required for the installation of localMDI package must be followed. Those are available on https://github.com/asutera/Local-MDI-importance/tree/main.

Once this is done, code is ready to be run from the folder containing the codes. TFE_Dyngen.py however needs as argument the # of the dyngen dataset tested.



