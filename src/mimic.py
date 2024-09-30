import os
import copy
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import Dataset
import glob
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F

def get_attr_max_min(attr):
    if attr == 'age':
        return 90, 18
    else:
        NotImplementedError

def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] *2 -1 #[-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            try:
                batch[k] = batch[k].float().unsqueeze(-1)
            except:
                batch[k] = batch[k]
    return batch
    

class MimicDataset(Dataset):
    def __init__(self, 
                 root, 
                 csv_file, 
                 transform=None, 
                 columns=None, 
                 concat_pa=True, 
                 use_only_pleural_effusion=True, 
                 create_bias=False, 
                 select_subgroup=False, 
                 race_choice=None, 
                 sex_choice=None, 
                 finding_choice=None,
                 only_no_finding=False
                 ):

        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.disease_labels = [
            'No Finding',
            'Other',
            'Pleural Effusion',
            'Lung Opacity'
        ]

        self.samples = {
            'age':[],
            'sex':[],
            'finding':[],
            'x':[],
            'race':[],
            'lung_opacity':[],
            'pleural_effusion':[],
            'dicom_id': [],
            'study_id': [],
            'path_preproc': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            if use_only_pleural_effusion and self.data.loc[idx, 'disease']=='Other':
                continue

            if only_no_finding and self.data.loc[idx, 'disease']!='No Finding':
                continue
            
            img_path = os.path.join(root, self.data.loc[idx, 'path_preproc'])
            
            lung_opacity = self.data.loc[idx, 'Lung Opacity']
            self.samples['lung_opacity'].append(lung_opacity)

            pleural_effusion = self.data.loc[idx, 'Pleural Effusion']
            self.samples['pleural_effusion'].append(pleural_effusion)

            disease = self.data.loc[idx, 'disease']
            finding = 0 if disease=='No Finding' else 1


            # Create a biased dataset
            if create_bias:
                if self.data.loc[idx, 'sex']=='Male' and finding==0:
                    continue
                if self.data.loc[idx, 'sex']=='Female' and finding==1:
                    continue

            # Select subgroup
            if select_subgroup:
                if self.data.loc[idx, 'sex']!=sex_choice and sex_choice is not None:
                    continue
                if self.data.loc[idx, 'race']!=race_choice and race_choice is not None:
                    continue
                if self.data.loc[idx, 'disease']!=finding_choice and finding_choice is not None:
                    # print(f"self.data.loc[idx, 'disease']: {self.data.loc[idx, 'disease']}")
                    continue
            
                    

            self.samples['path_preproc'].append(img_path)
            self.samples['x'].append(img_path)
            self.samples['finding'].append(finding)
            self.samples['age'].append(self.data.loc[idx, 'age'])
            self.samples['race'].append(self.data.loc[idx, 'race_label'])
            self.samples['sex'].append(self.data.loc[idx, 'sex_label'])
            self.samples['dicom_id'].append(self.data.loc[idx, 'dicom_id'])
            self.samples['study_id'].append(str(self.data.loc[idx, 'study_id']))

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa=concat_pa

    def __len__(self):
        return len(self.samples['x'])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}

        # print(f'sample before: {sample}')
        sample['x'] = imread(sample['x']).astype(np.float32)[None, ...]

        for k, v in sample.items():
            try:
                sample[k] = torch.tensor(v)
            except:
                sample[k] = sample[k]

        if self.transform:
            sample['x'] = self.transform(sample['x'])
        
        sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k in self.columns], dim=0)
        return sample

class MimicDataset_with_cfs(Dataset):
    def __init__(
        self, 
        csv_file, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        only_no_finding=False,
        which_cf='finding',
        target_cf=None,
        ):

        self.data = pd.read_csv(csv_file)
        self.race_category = ['White', 'Asian', 'Black']
        self.sex_category = ['Male', 'Female']
        self.finding_category = ['No_disease', 'Pleural_Effusion']
        self.all_category = {
                             'race': self.race_category,
                             'sex':self.sex_category,
                             'finding':self.finding_category,
                             }

        self.null_catgory = ['Null']
        self.transform = transform
        self.target_cf = target_cf
        self.disease_labels = [
            'No Finding',
            'Other',
            'Pleural Effusion',
            'Lung Opacity'
        ]

        self.samples = {
            'age':[],
            'sex':[],
            'finding':[],
            'x':[],
            'race':[],
            'dicom_id': [],
            'study_id': [],
            'path_preproc': [],
            'paths_cf': [],
            'path_cf_Null': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):

            if only_no_finding and self.data.loc[idx, 'finding']!=0:
                continue
            
            img_path = self.data.loc[idx, 'path_preproc']
            null_path = self.data.loc[idx, 'path_cf_Null']
            # Select subgroup
            if select_subgroup:
                if sex_choice is not None and self.data.loc[idx, 'sex']!=self.sex_category.index(sex_choice):
                    continue
                if race_choice is not None and self.data.loc[idx, 'race']!=self.race_category.index(race_choice):
                    continue
            
            paths_cf_sample = []

            for _which_cf in ['race','sex','finding']:
                if _which_cf not in which_cf:
                    continue
                for _cf_label in self.all_category[_which_cf]:
                    _path_cf = self.data.loc[idx, f'path_cf_{_cf_label}']
                    if _path_cf!='None':
                        paths_cf_sample.append(_path_cf)

            self.samples['path_preproc'].append(img_path)
            self.samples['x'].append(img_path)
            self.samples['finding'].append(self.data.loc[idx, 'finding'])
            self.samples['age'].append(self.data.loc[idx, 'age'])
            self.samples['race'].append(self.data.loc[idx, 'race'])
            self.samples['sex'].append(self.data.loc[idx, 'sex'])
            self.samples['dicom_id'].append(self.data.loc[idx, 'dicom_id'])
            self.samples['study_id'].append(str(self.data.loc[idx, 'study_id']))
            self.samples['paths_cf'].append(paths_cf_sample)
            self.samples['path_cf_Null'].append(null_path)

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex', 'finding']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa=concat_pa
        
    
    def __len__(self):
        return len(self.samples['x'])

    def __getitem__(self, idx):
        cf_sample = {k: v[idx] for k, v in self.samples.items()}
        sample = {k: v[idx] for k, v in self.samples.items()}
        sample['x'] = imread(sample['x']).astype(np.float32)[None, ...]

        if self.target_cf is None:
            # Randomly select a cf for augmentation
            _cf_path = random.sample(cf_sample['paths_cf'],1)[0]
        else:
            _cf_path = [_path for _path in cf_sample['paths_cf'] if self.target_cf in _path][0]

        print(f"_cf_path: {_cf_path}, self.target_cf: {self.target_cf}")
        cf_sample['x']= imread(_cf_path).astype(np.float32)[None, ...]

        for _k in self.race_category:
            if _k in _cf_path:
                _cf_label = 'race'
                _cf_value = int(self.race_category.index(_k))

        for _k in self.sex_category:
            if _k in _cf_path:
                _cf_label = 'sex'
                _cf_value = int(self.sex_category.index(_k))
        
        for _k in self.finding_category:
            if _k in _cf_path:
                _cf_label = 'finding'
                _cf_value = int(self.finding_category.index(_k))
        
        cf_sample[_cf_label] = _cf_value
        # Set cf_label, 1 for cf data, 0 for real data
        cf_sample['cf_label']=1
        sample['cf_label']=0

        # Get null batch
        null_sample = {k: v[idx] for k, v in self.samples.items()}
        null_sample['x'] = imread(sample['path_cf_Null']).astype(np.float32)[None, ...]  
        null_sample['cf_label']=1  

        for k, v in sample.items():
            try:
                sample[k] = torch.tensor(v)
                null_sample[k] = torch.tensor(null_sample[k] )
            except:
                sample[k] = sample[k]
                null_sample[k] = null_sample[k] 
        
        for k, v in cf_sample.items():
            try:
                cf_sample[k] = torch.tensor(v)
            except:
                cf_sample[k] = cf_sample[k]
        
        if self.transform:
            sample['x'] = self.transform(sample['x'])
            cf_sample['x'] = self.transform(cf_sample['x'])
            null_sample['x'] = self.transform(null_sample['x'])
        sample = norm(sample)
        cf_sample = norm(cf_sample)
        null_sample = norm(null_sample)
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k in self.columns], dim=0)
            null_sample['pa'] = torch.cat([null_sample[k] for k in self.columns], dim=0)
            cf_sample['pa'] = torch.cat([cf_sample[k] for k in self.columns], dim=0)
        return sample, cf_sample, null_sample

class MimicDataset_all_cfs(Dataset):
    def __init__(self, csv_file, transform=None, columns=None, concat_pa=True, select_subgroup=False, race_choice=None, sex_choice=None, only_no_finding=False):
        self.data = pd.read_csv(csv_file)
        self.race_category = ['White', 'Asian', 'Black']
        self.sex_category = ['Male', 'Female']
        self.transform = transform
        self.disease_labels = [
            'No Finding',
            'Other',
            'Pleural Effusion',
            'Lung Opacity'
        ]

        self.samples = {
            'age':[],
            'sex':[],
            'finding':[],
            'x':[],
            'race':[],
            'dicom_id': [],
            'study_id': [],
            'path_preproc': [],
            'paths_cf': [],
            'path_cf_Null': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):

            if only_no_finding and self.data.loc[idx, 'finding']!=0:
                continue
            
            img_path = self.data.loc[idx, 'path_preproc']
            null_path = self.data.loc[idx, 'path_cf_Null']
            # Select subgroup
            if select_subgroup:
                if self.data.loc[idx, 'sex']!=sex_choice and sex_choice is not None:
                    continue
                if self.data.loc[idx, 'race']!=race_choice and race_choice is not None:
                    continue
            
            paths_cf_sample = []
            for _cf_label in self.race_category+self.sex_category:
                _path_cf = self.data.loc[idx, f'path_cf_{_cf_label}']
                if _path_cf!='None':
                    paths_cf_sample.append(_path_cf)
        
            self.samples['path_preproc'].append(img_path)
            self.samples['x'].append(img_path)
            self.samples['finding'].append(self.data.loc[idx, 'finding'])
            self.samples['age'].append(self.data.loc[idx, 'age'])
            self.samples['race'].append(self.data.loc[idx, 'race'])
            self.samples['sex'].append(self.data.loc[idx, 'sex'])
            self.samples['dicom_id'].append(self.data.loc[idx, 'dicom_id'])
            self.samples['study_id'].append(str(self.data.loc[idx, 'study_id']))
            self.samples['paths_cf'].append(paths_cf_sample)
            self.samples['path_cf_Null'].append(null_path)

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa=concat_pa
    
    def __len__(self):
        return len(self.samples['x'])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}
        sample['x'] = imread(sample['x']).astype(np.float32)[None, ...]

        # Select all CFs for augmentation
        cf_batch = {}
        _cf_count=0
        for _cf_path in sample['paths_cf']:     
            for _k in self.race_category:
                if _k in _cf_path:
                    _cf_count+=1
                    _cf_label = 'race'
                    _cf_value = int(self.race_category.index(_k))
            
            # for _k in self.sex_category:
            #     if _k in _cf_path:
            #         _cf_count+=1
            #         _cf_label = 'sex'
            #         _cf_value = int(self.sex_category.index(_k))

            cf_batch[f'cf {_cf_count}']={k: v[idx] for k, v in self.samples.items()}
            cf_batch[f'cf {_cf_count}']['x']=imread(_cf_path).astype(np.float32)[None, ...]     
            cf_batch[f'cf {_cf_count}'][_cf_label]=_cf_value
            cf_batch[f'cf {_cf_count}']['cf_label']=1
        
        # Get null batch
        null_sample = {k: v[idx] for k, v in self.samples.items()}
        null_sample['x'] = imread(sample['path_cf_Null']).astype(np.float32)[None, ...]  
        null_sample['cf_label']=1  
        # Set cf_label, 1 for cf data, 0 for real data
        sample['cf_label']=0

        for k, v in sample.items():
            try:
                sample[k] = torch.tensor(v)
                null_sample[k] = torch.tensor( null_sample[k] )
            except:
                sample[k] = sample[k]
                null_sample[k] = null_sample[k] 

        for _cf_title in cf_batch.keys():
            for k, v in cf_batch[_cf_title].items():
                try:
                    cf_batch[_cf_title][k] = torch.tensor(v)
                except:
                    cf_batch[_cf_title][k] = cf_batch[_cf_title][k]
        
        if self.transform:
            sample['x'] = self.transform(sample['x'])
            null_sample['x'] = self.transform(null_sample['x'])
            for _cf_title in cf_batch.keys():
                cf_batch[_cf_title]['x'] = self.transform(cf_batch[_cf_title]['x'])
        sample = norm(sample)
        null_sample = norm(null_sample)
        for _cf_title in cf_batch.keys():
            cf_batch[_cf_title] = norm(cf_batch[_cf_title] )

        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k in self.columns], dim=0)
            null_sample['pa'] = torch.cat([null_sample[k] for k in self.columns], dim=0)
            for _cf_title in cf_batch.keys():
                cf_batch[_cf_title]['pa'] = torch.cat([cf_batch[_cf_title][k] for k in self.columns], dim=0)
        return sample, cf_batch, null_sample
        
if __name__=="__main__":
    data_dir = "/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/"
    csv_dir = "/vol/biomedic3/tx1215/chexpert-dscm/src/mimic_meta"

    csv_pd = pd.read_csv(csv_dir + '/mimic.sample.train.csv')
    d = glob.glob(csv_dir+'/*')

    train_set = MimicDataset(root=data_dir,
                            csv_file=os.path.join(csv_dir, 'mimic.sample.train.csv'),
                            transform=None,
                            columns=['age', 'race', 'sex', 'finding'],
                            concat_pa=True,
                            )

class MimicDataset_with_cfs_ratio(Dataset):
    def __init__(
        self, 
        csv_file, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        only_no_finding=False,
        which_cf='finding',
        target_cf=None,
        real_ratio=1.0,
        ):

        self.data = pd.read_csv(csv_file)
        self.race_category = ['White', 'Asian', 'Black']
        self.sex_category = ['Male', 'Female']
        self.finding_category = ['No_disease', 'Pleural_Effusion']
        self.all_category = {
                             'race': self.race_category,
                             'sex':self.sex_category,
                             'finding':self.finding_category,
                             }

        self.null_catgory = ['Null']
        self.transform = transform
        self.target_cf = target_cf
        self.disease_labels = [
            'No Finding',
            'Other',
            'Pleural Effusion',
            'Lung Opacity'
        ]

        self.samples = {
            'age':[],
            'sex':[],
            'finding':[],
            'x':[],
            'race':[],
            'dicom_id': [],
            'study_id': [],
            'path_preproc': [],
            'paths_cf': [],
            'path_cf_Null': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):

            if only_no_finding and self.data.loc[idx, 'finding']!=0:
                continue
            
            img_path = self.data.loc[idx, 'path_preproc']
            null_path = self.data.loc[idx, 'path_cf_Null']
            # Select subgroup
            if select_subgroup:
                if sex_choice is not None and self.data.loc[idx, 'sex']!=self.sex_category.index(sex_choice):
                    continue
                if race_choice is not None and self.data.loc[idx, 'race']!=self.race_category.index(race_choice):
                    continue
            
            paths_cf_sample = []
            # for _cf_label in self.race_category+self.sex_category:
            for _which_cf in ['race','sex','finding']:
                if _which_cf not in which_cf:
                    continue
                for _cf_label in self.all_category[_which_cf]:
                    _path_cf = self.data.loc[idx, f'path_cf_{_cf_label}']
                    if _path_cf!='None':
                        paths_cf_sample.append(_path_cf)

            self.samples['path_preproc'].append(img_path)
            self.samples['x'].append(img_path)
            self.samples['finding'].append(self.data.loc[idx, 'finding'])
            self.samples['age'].append(self.data.loc[idx, 'age'])
            self.samples['race'].append(self.data.loc[idx, 'race'])
            self.samples['sex'].append(self.data.loc[idx, 'sex'])
            self.samples['dicom_id'].append(self.data.loc[idx, 'dicom_id'])
            self.samples['study_id'].append(str(self.data.loc[idx, 'study_id']))
            self.samples['paths_cf'].append(paths_cf_sample)
            self.samples['path_cf_Null'].append(null_path)

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex', 'finding']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa=concat_pa
        
    
    def __len__(self):
        return len(self.samples['x'])

    def __getitem__(self, idx):
        cf_sample = {k: v[idx] for k, v in self.samples.items()}
        sample = {k: v[idx] for k, v in self.samples.items()}
        sample['x'] = imread(sample['x']).astype(np.float32)[None, ...]

        if self.target_cf is None:
            # Randomly select a cf for augmentation
            _cf_path = random.sample(cf_sample['paths_cf'],1)[0]
        else:
            _cf_path = [_path for _path in cf_sample['paths_cf'] if self.target_cf in _path][0]

        cf_sample['x']= imread(_cf_path).astype(np.float32)[None, ...]

        for _k in self.race_category:
            if _k in _cf_path:
                _cf_label = 'race'
                _cf_value = int(self.race_category.index(_k))

        for _k in self.sex_category:
            if _k in _cf_path:
                _cf_label = 'sex'
                _cf_value = int(self.sex_category.index(_k))
        
        for _k in self.finding_category:
            if _k in _cf_path:
                _cf_label = 'finding'
                _cf_value = int(self.finding_category.index(_k))
        
        cf_sample[_cf_label] = _cf_value
        # Set cf_label, 1 for cf data, 0 for real data
        cf_sample['cf_label']=1
        sample['cf_label']=0

        # Get null batch
        null_sample = {k: v[idx] for k, v in self.samples.items()}
        null_sample['x'] = imread(sample['path_cf_Null']).astype(np.float32)[None, ...]  
        null_sample['cf_label']=1  

        for k, v in sample.items():
            try:
                sample[k] = torch.tensor(v)
                null_sample[k] = torch.tensor(null_sample[k] )
            except:
                sample[k] = sample[k]
                null_sample[k] = null_sample[k] 
        
        for k, v in cf_sample.items():
            try:
                cf_sample[k] = torch.tensor(v)
            except:
                cf_sample[k] = cf_sample[k]
        
        if self.transform:
            sample['x'] = self.transform(sample['x'])
            cf_sample['x'] = self.transform(cf_sample['x'])
            null_sample['x'] = self.transform(null_sample['x'])
        sample = norm(sample)
        cf_sample = norm(cf_sample)
        null_sample = norm(null_sample)
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k in self.columns], dim=0)
            null_sample['pa'] = torch.cat([null_sample[k] for k in self.columns], dim=0)
            cf_sample['pa'] = torch.cat([cf_sample[k] for k in self.columns], dim=0)        
        return sample, cf_sample, null_sample