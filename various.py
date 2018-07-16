'''
Temporary file holding all necessary functions.
For a final pipeline version, group all functions into different relevant files.
'''
import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator

def gini_expo(actual, pred, expo):
    """Calculate gini coefficient, taking exposure into account."""
    #Create dataframe
    data_gini = pd.DataFrame(data = {'pred': pred, 'expo': expo,'actual': actual})

    #Order "Pred_data" by predicted response (decreasing)
    ord_pred_cum_expo = data_gini.sort_values(by=['pred'],ascending=False)

    #Add column for scaled cumulative sum of response
    z = pd.DataFrame(data = {'Cumulative_Actual_Response': ord_pred_cum_expo.actual.cumsum()})
    z["Cumulative_Actual_Response"] = z/z["Cumulative_Actual_Response"].tail(1).values
    ord_pred_cum_expo_actual = pd.concat([ord_pred_cum_expo, z], axis=1)

    #Calculate area under curve
    auc = (ord_pred_cum_expo_actual['expo']*(ord_pred_cum_expo_actual['Cumulative_Actual_Response']+ord_pred_cum_expo_actual['Cumulative_Actual_Response'].shift())/2/sum(ord_pred_cum_expo_actual['expo']))

    #Calculate gini coefficient
    gini = 2*(np.nansum(auc.values)-0.5)
    
    return gini

def gini(y_valid, y_pred):
    """Calculate gini coefficient."""
    assert y_valid.shape == y_pred.shape
    n_samples = y_valid.shape[0]
    
    # Sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_valid, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # Get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # Get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # Normalize to true Gini coefficient
    return G_pred / G_true


def gini_norm(y_valid, y_pred):
    """Calculate normalised gini coefficient."""
    return gini(y_valid, y_pred) / gini(y_valid, y_valid)


def xgb_gini(y_pred, y_true):
    """Calculate negative gini coefficient for xgb.fit method"""
    y_true = y_true.get_label()
    n_samples = y_true.shape[0]
    # Sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # Get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # Get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # Normalize to true Gini coefficient
    return ('negative_gini', -G_pred / G_true)


def variable_imp_function(data, y, pred, metric, model, feature):
    """Calculate variable importances."""
    # Compute initial metric
    init_metric = gini(y, pred)
    
    # Shuffling
    variable_imp = []
    variable_name = []
    for k in range(len(feature)):
        data_shuffle =  pd.DataFrame(data)
        data_shuffle.iloc[:, k] = data_shuffle.sample(frac=1).iloc[:, k].values
        data_shuffle = data_shuffle.as_matrix()
        
        pred_shuffle = model.predict(data_shuffle)
        shuffle_metric =  init_metric - gini(y, pred_shuffle)
        variable_imp.append(shuffle_metric)
        variable_name.append(feature[k])
    
    final_var = pd.DataFrame(
        {'Name': variable_name,
         'Imp': variable_imp
        })
    
    return final_var


def variable_importances(data, y, model, metric, n_iter=3):
    """Compute variable importances based on a model and passed metric
    data -- pandas dataset with of predictor columns
    y -- pandas series of response column
    model -- sklearn classifier
    metric -- metric function with (actual, predicted) arguments
    """
    
    # Compute initial metric
    original_preds = model.predict(data)
    init_metric = metric(y, original_preds)

    # Shuffle each col and compare gini
    variable_imp = []
    variable_name = []
    for col in data.columns:
        data_shuffle = data.copy()
        shuffle_metric = 0
        for i in range(n_iter):
            data_shuffle.loc[:, col] = data_shuffle.sample(frac=1).loc[:, col].reset_index(drop=True)
            pred_shuffle = model.predict(data_shuffle)
            shuffle_metric +=  init_metric - metric(y, pred_shuffle)
            
        shuffle_metric = shuffle_metric / n_iter
        variable_imp.append(shuffle_metric)
        variable_name.append(col)
        
    final_var = pd.DataFrame({'Name': variable_name, 'Imp': variable_imp}).sort_values('Imp', ascending=False).reset_index(drop=True)
    return final_var


def xgb_create_monotone_string(colname, negative_monotonicity, positive_monotonicity):
    """Generate xgboost monotonicity list."""
    mono_vector = []
    for k in colname:
            if k in positive_monotonicity:
                mono_vector.append(1)
            elif  k in negative_monotonicity:
                mono_vector.append(-1)
            else:
                 mono_vector.append(0)
                    
    return mono_vector


def mono_to_string(mono_list):
    """Transform monotonicity list into formatted string."""
    return str(mono_list).replace('[', '(').replace(']', ')')


def lift_chart(pred, expo, bins):
    """Generate lift chart data."""
    data_lift = pd.DataFrame({'Pred': pred, 'Expo': expo})
    return data_lift


def get_globals():
    """Return user defined global variables."""
    with open('/data/dss_data/local/variables.json', 'r') as f:
        data = json.loads(f.read())
    return data


def set_globals(var):
    """
    Adds the dictionary to the global variable file.
    """
    assert isinstance(var, dict), '%r is not a dict' % var
    old_vars = get_globals()
    old_vars.update(var)
    with open('/data/dss_data/local/variables.json', 'w') as f:
        f.write(json.dumps(old_vars))


def clear_globals():
    """Clear global variable dictionary."""
    with open('/data/dss_data/local/variables.json', 'w') as f:
        f.write('{}')


class OrdinalEncoder(BaseEstimator):
    """Transforms features using ordinal encoding (ascending order)."""
    def __init__(self):
        pass
    
    def fit_transform(self, X, y=None):
        #return X.apply(lambda col: col.map(col.value_counts()))
        return self.fit(X).transform(X).astype('int32')
    
    def fit(self, X, y=None):
        self.key_ = {}
        for col in X:
            freqs = X[col].value_counts()
            self.key_[col] = dict(zip(freqs.index, np.argsort(freqs.values)))
        return self
        
    def transform(self, X, y=None):
        return X.apply(lambda col: col.map(self.key_[col.name])).astype('int32')
