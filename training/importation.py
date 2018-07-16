import pandas as pd
import numpy as np

def importation_data(input_data_path):
    """Import data set
    input_data_path -- string path
    """
    #Import
    dataset = pd.read_csv(input_data_path, index_col=0)
    return dataset
