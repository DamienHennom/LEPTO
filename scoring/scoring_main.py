from training.BP01.bp01_main import bp01_scoring  

def scoring_bp(model,dataset,input_prj,bp="bp01"):
    if bp=="bp01":
        pred = bp01_scoring(model,dataset,input_prj)
    return pred
