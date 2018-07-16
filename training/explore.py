import pandas as pd
import numpy as np
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'figure.max_open_warning': 0})
import general_settings
from training.func_use import unique

def explore_data(data,prj_info,TMP=1234):
     """Explore a dataset
     dataset -- pandas dataset
     prj_info -- dictionnary containing projet information (response...)
     """
     print("   Data file rows and columns are : ", data.shape)
     #Open pdf
     pp = PdfPages(prj_info['OUTPUT_PATH'] + "exploration_" + str(TMP) + ".pdf")

     #Plot average
     plot_average_reponse(data,prj_info,pp,TMP)

     #Close pdf
     pp.close()
     return None

def plot_average_reponse(data,prj_info,pp=PdfPages("exploration.pdf"),TMP=1234,bins=20):
    """Plot average response for all variables in dataset and save plot in pdf
    dataset -- pandas dataset
    prj_info -- dictionnary containing projet information (response...)
    """
    #Copy data
    data = data.copy()
    #Slice data
    data = data.sample(n = min(10000,data.shape[0]),random_state=1234)
    #Colnames
    var_to_plot = list(set(data.columns.values)-set(prj_info['PRJ_COLUMN'].values()))

    #Loop figure
    pbar = ProgressBar()
    for var in pbar(var_to_plot):
        #Bins
        if data[var].dtype.name != "category" and len(data[var].unique())>bins:
            data["var_new"] = pd.qcut(data[var], bins, duplicates='drop')
        else:
            data["var_new"] = data[var].astype(str)
        data_plot = data.groupby("var_new").agg({prj_info['PRJ_COLUMN']["RESPONSE"]: 'mean', "var_new": 'count'})

        #Table
        data_plot = data.groupby("var_new").agg({prj_info['PRJ_COLUMN']["RESPONSE"]: 'mean', "var_new": 'count'})

        #Build plot
        f, ax = plt.subplots()
        ax2 =ax.twinx()
        sns.barplot(x=data_plot.index.tolist(), y="var_new",data=data_plot,ax=ax, color="dodgerblue")
        sns.pointplot(x=data_plot.index.tolist(), y=prj_info['PRJ_COLUMN']["RESPONSE"], data=data_plot,ax=ax2, color="chartreuse")
        ax.set_xlabel(var)
        ax.set_ylabel(var)
        ax2.set_ylabel(prj_info['PRJ_COLUMN']["RESPONSE"])
        plt.title("Average reponse by " + var)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
        
        pp.savefig(f)

    return None
