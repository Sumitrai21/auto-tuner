import os
import shutil




class create_dataset(path,dst_path):
    def __init__(self):
        self.path = path
        self.dst_path = dst_path
        self.files_list = None


    def check_new_data(self):
        if len(os.listdir(self.path))> 0:
            return True

        else:
            return False

    def locate_file(self,i):
        if i.split('.'[-1]) == 'txt':
            src_path = self.path+"/"+'i'
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
                locate_file(i)
            

        else:
            print('No new files. Model is up to date')
