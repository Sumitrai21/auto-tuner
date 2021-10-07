import os
import shutil
#from runner import RANK
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
    def __init__(self,cfg):
        self.cfg = cfg
        self.RANK = int(os.getenv('RANK', -1))
    
    def get_dataloader(self):
        trainloader,dataset = create_dataloader('dataset/train',312,32,
                                32,True,hyp='data/hyps/hyp.scratch.yaml',augment=self.cfg.Training.augment,
                                cache=False,rect=False,rank=self.RANK, workers=8,
                                image_weights=False,quad=False,prefix=colorstr('train: '))

        return trainloader,dataset


if __name__ == '__main__':
    print('running this file')
    a = CreateDataloader()
    trainloader,_ = a.get_dataloader()