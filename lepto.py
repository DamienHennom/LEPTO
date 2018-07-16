############################################################################################################
############################################################################################################

                         #         #######   #######   #######     #####
                         #         #         #     #      #       #     #
                         #         #######   #######      #      #       #
                         #         #         #            #       #     #
                         #######   #######   #            #        #####

#############################################################################################################
#############################################################################################################

########################################################
# LIBRARY
########################################################

import general_settings
from training.importation import importation_data
from training.explore import explore_data
from training.prepare import prepare_dataset
from training.analysis import analyse_model
from training.clean import clean_data
from training.savepipeline import save_pipeline_zip
from training.BP01.bp01_main import bp01_model
from training.BP02.bp02_main import bp02_model
from training.BP03.bp03_main import bp03_model
from scoring.scoring_main import scoring_bp
from training.environement import TMP
import os

########################################################
# FUNCTIONS
########################################################

def training_pilot(input_prj):
    """Training pilot
    input_prj -- dictionnary containing projet information (response...)
    """
    #Save code
    print('Save Code')
    save_pipeline_zip(input_prj['CODE_DIRECTORY'],input_prj['OUTPUT_PATH'],TMP)
    #Import Data
    print('Import Dataset')
    dataset = importation_data(input_prj['INPUT_PATH_DATA'])
    #Clean Data
    print('Clean Dataset')
    dataset = clean_data(dataset)
    #Explore Dataset
    print('Explore Dataset')
    #explore_data(dataset,general_settings.PRJ_SET,TMP)
    #Prepare Dataset
    print('Prepare Dataset')
    train,test = prepare_dataset(dataset,input_prj['PRJ_COLUMN'])
    #Run BP
    print('Run blueprint')
    #model,pred_fold,pred_test,variables_selected = bp01_model(train,test,input_prj,TMP)
    model,pred_fold,pred_test,variables_selected = bp03_model(train,test,input_prj,TMP)
    #model,pred_fold,pred_test,variables_selected = bp02_model(train,test,input_prj,TMP)
    #Run Analysis
    print('Analysis')
    #train_final, test_final = analyse_model(train,test,input_prj,model,pred_fold,pred_test,variables_selected,TMP)
    return model,pred_fold,pred_test,variables_selected
    
def scoring_pilot(model,dataset,input_prj,bp="bp01"):
    """Scoring pilot
    model -- model created via training_pilot
    dataset -- pandas dataframe
    input_prj -- dictionnary containing projet information (response...)
    bp -- string
    """
    #Clean Data
    print('Clean Dataset')
    dataset = clean_data(dataset)
    #Score on new data 
    print('Score data')
    pred = scoring_bp(model,dataset,input_prj,"bp01")
    return pred 

if __name__ == '__main__':
    training_pilot(general_settings.PRJ_SET)
    
