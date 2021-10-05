import os
import shutil
from utils.datasets import create_dataloader
from utils.general import colorstr




class CreateDataset():
    def __init__(self,cfg):
        self.path = cfg.Paths.src_path
        self.dst_path = cfg.Paths.dst_path
        self.files_list = None


    def check_new_data(self):
        if len(os.listdir(self.path))> 0:
            return True

        else:
            return False

    def locate_file(self,i):
        if i.split('.'[-1]) == 'txt':
            src_path = self.path+"/"+'i' #os.path.join()
            new_path = self.dst_path+'/'+'labels'+'/'+i
            shutil.move(src_path,new_path)

        elif i.split('.')[-1] == '.jpg':
            src_path = self.path+"/"+'i'
            new_path = self.dst_path+'/'+'images'+'/'+i
            shutil.move(src_path,new_path)

        else:
            print('File Format not supported')


    def separate_files(self):
        if self.check_new_data():
            self.files_list = os.listdir(self.path)
            for i in self.files_list:
                self.locate_file(i)
            

        else:
            print('No new files. Model is up to date')


class CreateDataloader():
    def __init__(self,cfg,RANK):
        self.cfg = cfg
        self.RANK = RANK
    
    def get_dataloader(self):
        trainloader,dataset = create_dataloader(self.cfg.Paths.train_pth,self.cfg.Training.imgsz,self.cfg.Training.batch_size,
                                self.cfg.Training.gs,self.cgf.Training.single_cls,hyp=self.cfg.Training.hyp,augment=self.cfg.Training.augment,
                                cache=self.cfg.Training.chache_images,rect=self.cfg.Training.rect,rank=self.RANK, workers=self.cfg.Training.workers,
                                image_weights=self.cfg.Training.image_weights,quad=self.cfg.Training.quad,prefix=colorstr('train: '))

        return trainloader,dataset
