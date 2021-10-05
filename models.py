import os
import tqdm
import torch 
import mlflow
import mlflow.pytorch
import liteconfig

from pathlib import Path

cfg = liteconfig.Config('config.ini')

ML_FLOW_URL = cfg.Mlflow.mlflow_url
mlflow.set_tracking_uri(ML_FLOW_URL)

def load_model():
    model_name = cfg.Mlflow.model_name
    model_version = cfg.Mlflow.model_version
    if os.path.isfile(f'{cfg.Detection.weights}.pt'):
        pass

    else:
        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
        torch.save(model,f'{cfg.Detection.weights_savename}.pt')


    model = torch.hub.load(os.getcwd(), 'custom', path=f'{cfg.Detection.weights}.pt', source='local', force_reload = True)
    print('model loaded')
    return model


print(load_model())