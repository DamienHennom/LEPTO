import pandas as pd
import numpy as np

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
