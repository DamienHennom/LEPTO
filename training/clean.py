import pandas as pd
import numpy as np


def clean_data(data):
    """Clean dataset
    data -- pandas dataset
    """
    #Remove constant column
    data = data.loc[:, (data != data.iloc[0]).any()].copy()
    #Convert string to categorical
    categorical_feature = data.select_dtypes(include=['object']).columns.values
    for col in categorical_feature:
        data[col] = data[col].astype('category')
    return data
