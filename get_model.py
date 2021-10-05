#@uthor-Sumit

#import the depedencies
import os
import tqdm
import torch 
import mlflow
import mlflow.pytorch
import liteconfig

from pathlib import Path


class GetModel():
    def __init__(self,cfg):
        
        self.cfg = cfg
        self.model_name = cfg.Mlflow.model_name
        self.model_version = cfg.Mlflow.model_version
        self.ML_FLOW_URL = cfg.Mlflow.mlflow_url
        self.model =None


    def load_model(self):
        #loads the model from mlflow server
        if os.path.isfile(f'{cfg.Detection.weights}.pt'):
            pass

        else:
            self.model = mlflow.pytorch.load_model(model_uri=f"models:/{self.model_name}/{self.model_version}")
        
        self.model = torch.hub.load(os.getcwd(),'custom',path=f'{self.cfg.Detection.weights}.pt',source='local',force_reload=True)
        print(f"Yolo {self.model_name}/{self.model_version} loaded...")












