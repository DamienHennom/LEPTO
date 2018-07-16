import pandas as pd
import numpy as np
from training.func_use import error_metric
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from matplotlib.backends.backend_pdf import PdfPages
from training.func_use import variable_importances
from training.func_use import build_data
from training.func_use import build_data_one
from training.func_use import data_feature_selected
from training.func_use import data_feature_selected_one
from training.func_use import unique
from numpy import inf
from itertools import compress
from progressbar import ProgressBar

def analyse_model(train,test,input_prj,model,pred_fold,pred_test,variables_selected,TMP=1234):
    """Analyse a model
    train -- pandas dataframe
    test -- pandas dataframe
    input_prj -- dictionnary containing projet information (response...)
    pred_fold -- pandas dataframe
    pred_test -- pandas dataframe
    variables_selected -- list variables in model
    """  
    #Open pdf
    pp = PdfPages(input_prj['OUTPUT_PATH'] + "analysis_" + str(TMP) + ".pdf")
    #Prepare Data
    train_an, test_an = prepare_data_analysis(train,test,input_prj,model,pred_fold,pred_test)
    #Performance
    if input_prj['PRJ_COLUMN']['EXPOSURE'] != None :
        exposure_train_value = train_an[input_prj['PRJ_COLUMN']['EXPOSURE']]
        exposure_test_value = test_an[input_prj['PRJ_COLUMN']['EXPOSURE']]
    else :
        exposure_train_value = None
        exposure_test_value = None
    metric_pred_cv = error_metric(train_an[input_prj['PRJ_COLUMN']['RESPONSE']],
                                  train_an["Pred"],
                                  exposure_train_value,
                                  input_prj['METRIC'])
    metric_test = error_metric(test_an[input_prj['PRJ_COLUMN']['RESPONSE']],
                               test_an["Pred"],
                               exposure_test_value,
                               input_prj['METRIC'])
    print("    Fold " + input_prj['METRIC'] + " : " + str(metric_pred_cv))
    print("    Test " + input_prj['METRIC'] + " : " + str(metric_test))
    #Variable importance
    print("    Variable importance")
    plot_var = variable_importance_plot(train,input_prj,model,variables_selected)
    pp.savefig(plot_var)
    print("    Lift chart")
    plot_lift = lift_chart(train_an,input_prj['PRJ_COLUMN']["RESPONSE"],"Pred")
    pp.savefig(plot_lift)
    #Marginal effect
    print("    Marginal effect")
    marginal_effect(train,pred_fold,input_prj,model,variables_selected,pp)
    #Close Pdf
    pp.close()
    return train_an, test_an

def marginal_effect(train,pred,input_prj,model,variables_selected,saveplot,bins=20):
    """Marginal effect
    train -- pandas dataframe
    input_prj -- dictionnary containing projet information (response...)
    pred -- pandas dataframe
    model -- sklearn model
    variables_selected -- list variables in model
    saveplot -- PdfPages
    """  
    #Select var
    train = data_feature_selected_one(train,variables_selected,list(input_prj['PRJ_COLUMN'].values()))
    #Slice data
    train_sample = train.sample(n = min(10000,train.shape[0]),random_state=1234)
    #Create Dataset
    y_train,X_train,W_train,O_train,monotonicity_vec = build_data_one(train_sample,input_prj)
    #Marginal
    variables = X_train.columns.values
    pbar = ProgressBar()
    for var in pbar(variables):
        X_train_used = X_train.copy()
        #Unique value
        if X_train_used[var].dtype.name != "category" and len(X_train_used[var].unique())>bins:
            unique_value = list(X_train_used[var].quantile(list(np.arange(0.0, 1.0, 0.05))))
            unique_value = unique([round(x,3) for x in unique_value])
        else:
            unique_value = X_train_used[var].unique()
            unique_value = [x for x in unique_value if str(x) != 'nan']
        #Loop on unique value
        data_plot = pd.DataFrame([])
        value_before = -inf
        unique_value = sorted(unique_value)
        for value in unique_value:
            #Change value
            if isinstance(value, str):
                X_train_used[var] = value
                X_train_used[var] = X_train_used[var].astype('category')
                mean_value_actual = np.nanmean(train.loc[train[var] == value,input_prj['PRJ_COLUMN']["RESPONSE"]].tolist())
                mean_value_pred = np.nanmean(list(pred[pd.Series(train[var]) == value]["Pred"]))
                count_value = len(list(pred[pd.Series(train[var]) == value]["Pred"]))
            else:
                X_train_used[var] = value
                mean_value_actual = np.nanmean(train.loc[(train[var] <= value)  & (train[var] > value_before),input_prj['PRJ_COLUMN']["RESPONSE"]].tolist())
                mean_value_pred = np.nanmean(list(pred[(pd.Series(train[var]) <= value) & (pd.Series(train[var]) > value_before)]["Pred"]))
                count_value = len(list(pred[(pd.Series(train[var]) <= value) & (pd.Series(train[var]) > value_before)]["Pred"]))
            #Prediction
            pred_marginal_mean = np.mean(model.predict(X_train_used))
            result_frame = pd.DataFrame({var: [value], 'Marginal': [pred_marginal_mean], 'Actual' : [mean_value_actual], 'Pred' : [mean_value_pred],'Count' : [count_value]})
            data_plot = data_plot.append(result_frame)

            value_before = value
        #Plot
        f, ax = plt.subplots()
        ax2 =ax.twinx()
        sns.barplot(x=var, y='Count',data=data_plot,ax=ax, color="dodgerblue")
        sns.pointplot(x=var, y='Actual', data=data_plot,ax=ax2, color="chartreuse", label="Actual")
        sns.pointplot(x=var, y='Pred', data=data_plot,ax=ax2, color="orange", label="Prediction")
        sns.pointplot(x=var, y='Marginal', data=data_plot,ax=ax2, color="black", label="Marginal")
        ax.set_xlabel(var)
        ax.set_ylabel('Count')
        ax2.set_ylabel('Average')
        ax2.legend(handles=ax2.lines[::len(data_plot)+1], labels=["Actual","Prediction","Marginal"])
        plt.title("Marginal effect " + var)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
        saveplot.savefig(f)
    return None

def lift_chart(data,response,pred,bins=20):
    """Lift chart
    data -- pandas dataframe
    response -- string response column
    pred -- string pred column
    """  
    #Sort
    data = data.sort_values(pred, ascending=[1])
    #Create bin
    data["bin_lift"] = pd.cut(range(data.shape[0]), bins)
    #Group by
    data_plot = data.groupby("bin_lift").agg({response: 'mean', pred: 'mean'})
    data_plot["bin_lift"] = range(1,data_plot.shape[0]+1)
    #Plot
    fig, ax = plt.subplots()
    sns.pointplot(x='bin_lift', y=response, data=data_plot, ax=ax, color='dodgerblue', label="Response")
    sns.pointplot(x='bin_lift', y=pred, data=data_plot, ax=ax, color='chartreuse', label="Prediction")
    ax.set_xlabel("Bins")
    ax.set_ylabel("Average")
    ax.legend(handles=ax.lines[::len(data_plot)+1], labels=["Response","Prediction"])
    plt.title("Lift chart")
    return fig

def prepare_data_analysis(train,test,input_prj,model,pred_fold,pred_test):
    """Join data for analysis
    train -- pandas dataframe
    test -- pandas dataframe
    input_prj -- dictionnary containing projet information (response...)
    model -- sklearn model
    pred_fold -- pandas dataframe
    pred_test -- pandas dataframe
    """  
    #Join data and pred
    train_an = pd.merge(train, pred_fold, how='left', on=input_prj['PRJ_COLUMN']['INDEX'])
    test_an = pd.merge(test, pred_test, how='left', on=input_prj['PRJ_COLUMN']['INDEX'])
    #Select column
    selected_feature = [list(input_prj['PRJ_COLUMN'].values()),["Pred"]]
    selected_feature = [item for sublist in selected_feature for item in sublist]
    selected_feature = [x for x in selected_feature if x is not None]
    train_an = train_an[selected_feature]
    test_an = test_an[selected_feature]
    return train_an, test_an

def variable_importance_plot(train,prj_info,model,variables_selected):
    """Plot variable importance shuffling
    train -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    model -- sklearn model
    variables_selected -- list variable selected
    """  
    #Select var
    train = data_feature_selected_one(train,variables_selected,list(prj_info['PRJ_COLUMN'].values()))
    #Create Dataset
    y_train,X_train,W_train,O_train,monotonicity_vec = build_data_one(train,prj_info)
    #Variable imp
    variable_data = variable_importances(X_train, y_train,W_train, model, prj_info['METRIC'])
    #Plot
    fig, ax = plt.subplots()
    ax = sns.barplot(x="Imp", y="Name", data=variable_data, color="dodgerblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Variable")
    plt.title("Variable importance (Compute on train data)")
    return fig

