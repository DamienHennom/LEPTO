from metric.metric import gini_expo
from metric.metric import gini
import pandas as pd
from sklearn.metrics import roc_auc_score

def create_monotone_string(colname, negative_monotonicity, positive_monotonicity):
    """Create a string for XGB, Lightgbm monotonicity constraint
    colname -- list colnames
    negative_monotonicity -- list of colnames 
    positive_monotonicity -- list of colnames 
    """
    mono_vector = []
    for k in colname:
            if k in positive_monotonicity:
                mono_vector.append(1)
            elif  k in negative_monotonicity:
                mono_vector.append(-1)
            else:
                 mono_vector.append(0)
                              
    return mono_vector

def error_metric(y_valid, y_pred,expo,metric):
    """Error metric with string
    y_valid -- pandas series 
    y_pred -- pandas series 
    expo -- pandas series
    metric -- string 
    """
    if metric == "gini":
        error = gini(y_valid, y_pred)
    if metric == "gini_expo":
        error = gini_expo(y_valid, y_pred,expo)
    if metric == "auc":
        error =  roc_auc_score(y_valid, y_pred)
    
    return error

def variable_importances(data, y, expo,model, metric, n_iter=1):
    """Variable importance shuffling
    data -- pandas dataframe
    y -- pandas series response
    expo -- pandas series expo
    metric -- function metric
    """
    # Compute initial metric
    original_preds = model.predict(data)
    init_metric = error_metric(y, original_preds,expo,metric)

    # Shuffle each col and compare gini
    variable_imp = []
    variable_name = []
    for col in data.columns:
        data_shuffle = data.copy()
        shuffle_metric = 0
        for i in range(n_iter):
            data_shuffle.loc[:, col] = data_shuffle.sample(frac=1).loc[:, col].reset_index(drop=True)
            pred_shuffle = model.predict(data_shuffle)
            shuffle_metric +=  init_metric - error_metric(y, pred_shuffle,expo,metric)
            
        shuffle_metric = shuffle_metric / n_iter
        variable_imp.append(shuffle_metric)
        variable_name.append(col)
        
    final_var = pd.DataFrame({'Name': variable_name, 'Imp': variable_imp}).sort_values('Imp', ascending=False).reset_index(drop=True)
    return final_var


def build_data(train,test,prj_info):
    """Build data for modelling
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    """
    #Train
    y_train,X_train,W_train,O_train,monotonicity_vec_train = build_data_one(train,prj_info)
    #Test
    y_test,X_test,W_test,O_test,monotonicity_vec_test = build_data_one(test,prj_info)
    
    return y_train,y_test,X_train,X_test,W_train,W_test,O_train,monotonicity_vec_train


def build_data_one(data,prj_info):
    """Build data for modelling
    data -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    """
    #Y
    y = data[prj_info['PRJ_COLUMN']["RESPONSE"]].values
    #X
    predictors = list(set(data.columns.values)-set(prj_info['PRJ_COLUMN'].values()))
    X = data[predictors]
    #W
    if prj_info['PRJ_COLUMN']['WEIGHT'] != None or prj_info['PRJ_COLUMN']['EXPOSURE'] != None:
        if prj_info['PRJ_COLUMN']['WEIGHT'] != None and prj_info['PRJ_COLUMN']['EXPOSURE'] == None:
            W = data[prj_info['PRJ_COLUMN']['WEIGHT']].values
        elif prj_info['PRJ_COLUMN']['WEIGHT'] == None and prj_info['PRJ_COLUMN']['EXPOSURE'] != None:
            W = data[prj_info['PRJ_COLUMN']['EXPOSURE']].values
    else:
        W = None
    #O
    if prj_info['PRJ_COLUMN']['OFFSET'] != None:
        O = data[prj_info['PRJ_COLUMN']['OFFSET']].values
    else:
        O = None

    #Monotonicity
    monotonicity_vec = create_monotone_string(X.columns.values,prj_info['NEGATIVE_MONO'],prj_info['POSITIVE_MONO'])

    return y,X,W,O,monotonicity_vec

def feature_select(X_test, y_test,expo, model,metric, n_iter=1):
    """Variable importance selection
    X_test -- pandas dataframe (predictors)
    y_test -- pandas series (reponse)
    expo -- pandas series expo
    metric -- function metric
    """
    #All variables
    variable_data = variable_importances(X_test, y_test,expo,model, metric,n_iter)
    #Only feature selected
    variable = variable_data.loc[variable_data['Imp'] > 0]["Name"].tolist()
    return variable,variable_data

def data_feature_selected(train,test,variables_selected,special_col):
    """Variable selection creation of data
    train -- pandas dataframe 
    test -- pandas dataframe 
    variables_selected -- list of variables
    special_col -- special column to keep
    """
    train = data_feature_selected_one(train,variables_selected,special_col)
    test = data_feature_selected_one(test,variables_selected,special_col)
    return train,test

def data_feature_selected_one(data,variables_selected,special_col):
    """Variable selection creation of data
    data -- pandas dataframe 
    test -- pandas dataframe 
    variables_selected -- list of variables
    special_col -- special column to keep
    """
    #Copy data
    data = data.copy()
    #All variable selected
    all_var = variables_selected+special_col
    all_var = [x for x in all_var if x is not None]
    #Select
    data = data[all_var]

    return data

def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
