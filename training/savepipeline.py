import shutil

def save_pipeline_zip(inputhpath,outputpath,TMP=1234):
    output_filename = outputpath + 'CODE_' + str(TMP)
    shutil.make_archive(output_filename, 'zip', inputhpath)
    
