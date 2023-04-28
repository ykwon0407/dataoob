import pandas as pd
import numpy as np
import os, pickle, argparse
import openml

def create_df_openml_datasets(n_minimum=5000, n_maximum=100000, p_maximum=1000):
    '''
    Return pandas.DataFrame of available datasets such that 
    - no missing values
    - 5000 <= sample size <= 100000
    - input dimension <= 1000
    '''
    df_every_datasets = openml.datasets.list_datasets(output_format="dataframe")

    result = []
    for index, row in df_every_datasets.iterrows():
        if (row['NumberOfInstancesWithMissingValues'] != 0) or (row['NumberOfMissingValues'] != 0):
            continue
        if (row['NumberOfInstances'] < n_minimum) or (row['NumberOfInstances'] > n_maximum):
            continue
        if row['NumberOfFeatures'] > p_maximum:
            continue
        
        if row['NumberOfClasses'] >= 2:
            result.append([row['did'], row['name'], 'clf'])
        elif row['NumberOfClasses'] == 0:
            result.append([row['did'], row['name'], 'reg'])
        else:
            pass
            
    df_openml_datasets = pd.DataFrame(result, columns=['dataset_id', 'name', 'task_type'])
    return df_openml_datasets

def judge(X_col):
    for i in X_col.index.values:
        if X_col[i] is not None:
            return isinstance(X_col[i], str)
    return False

def dataset_maker(dataset_num, name, task_type):
    save_dict = {}
    try:
        dataset = openml.datasets.get_dataset(dataset_num)
    except:
        return {'dataset_name':'no name'}, 'no name', 'cannot get dataset', 'no name'
    save_dict['description'] = dataset.description
    save_dict['dataset_name'] = name
    file_name = name+'_'+str(dataset_num)+'.pkl'
    
    try:
        X, y, categorical_indicator, attribute_info = dataset.get_data(target=dataset.default_target_attribute,
                                                                       dataset_format="dataframe")
    except:
        return save_dict, file_name, 'attribute error because of openml', save_dict['dataset_name']
    
    if task_type in ['clf']:
        list_of_classes, y = np.unique(y, return_inverse=True)
        
    if y is None:
        return save_dict, file_name, 'y is none', save_dict['dataset_name']
    
    row_num = X.shape[0] 
    y = np.array(y)
    
    save_dict['y'] = y
    save_dict['flag'] = -1000
    save_dict['num_names'] = []
    save_dict['cat_names'] = []
    save_dict['X_num'] = []
    save_dict['X_cat'] = []
    save_dict['n_samples'] = row_num
    missing_val = 0
    for name_ind, name in enumerate(attribute_info):
        if name in ['ID','url']:
            continue
            
        if (categorical_indicator[name_ind]) or (judge(X[name])):
            save_dict['cat_names'].append(attribute_info[name_ind])
            list_of_classes, now = np.unique(X[name].astype(str), return_inverse=True)
            for j_ind, j in enumerate(X[name].index.values):
                if isinstance(X[name][j], float):
                    now[j_ind] = save_dict['flag']
                    missing_val += 1
            save_dict['X_cat'].append(now)
        else:
            save_dict['num_names'].append(attribute_info[name_ind])
            save_dict['X_num'].append(np.array(X[name], dtype=float))
            
    save_dict['X_num'] = np.array(save_dict['X_num']).transpose()
    save_dict['X_cat'] = np.array(save_dict['X_cat']).transpose()
    save_dict['missing_value'] = len(save_dict['X_num'][np.isnan(save_dict['X_num'])]) + len(save_dict['X_cat'][np.isnan(save_dict['X_cat'])]) + missing_val
    save_dict['X_num'][np.isnan(save_dict['X_num'])] = save_dict['flag']
    save_dict['X_cat'][np.isnan(save_dict['X_cat'])] = save_dict['flag']
    
    return save_dict, file_name, 'success', save_dict['dataset_name']



