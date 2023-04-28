import numpy as np
import pandas as pd
import pickle

def load_data(problem, dataset, **dargs):
    print('-'*30)
    print(dargs)
    if problem=='clf':
        (X, y), (X_val, y_val), (X_test, y_test)=load_classification_dataset(dataset=dataset,
                                                                            n_data_to_be_valued=dargs['n_data_to_be_valued'],
                                                                            n_val=dargs['n_val'],
                                                                            n_test=dargs['n_test'],
                                                                            input_dim=dargs.get('input_dim', 10),
                                                                            clf_path=dargs.get('clf_path'),
                                                                            openml_path=dargs.get('openml_clf_path'))
        if dargs['is_noisy'] > 0:
            n_class=len(np.unique(y))

            # training is flipped
            flipped_index=np.random.choice(np.arange(dargs['n_data_to_be_valued']), 
                                           int(dargs['n_data_to_be_valued']*dargs['is_noisy']), 
                                           replace=False) 
            random_shift=np.random.choice(n_class-1, len(flipped_index), replace=True)
            y[flipped_index]=(y[flipped_index] + 1 + random_shift) % n_class

            # validation is also flipped
            flipped_val_index=np.random.choice(np.arange(dargs['n_val']),
                                               int(dargs['n_val']*dargs['is_noisy']), 
                                               replace=False) 
            random_shift=np.random.choice(n_class-1, len(flipped_val_index), replace=True)
            y_val[flipped_val_index]=(y_val[flipped_val_index] + 1 + random_shift) % n_class 
            return (X, y), (X_val, y_val), (X_test, y_test), flipped_index
        else:
            return (X, y), (X_val, y_val), (X_test, y_test), None
    else:
        raise NotImplementedError('Check problem')

def load_classification_dataset(dataset,
                                n_data_to_be_valued, 
                                n_val, 
                                n_test, 
                                input_dim=10,
                                clf_path='clf_path',
                                openml_path='openml_path'):
    '''
    This function loads classification datasets.
    n_data_to_be_valued: The number of data points to be valued.
    n_val: Validation size. Validation dataset is used to evalute utility function.
    n_test: Test size. Test dataset is used to evalute model performance.
    clf_path: path to classification datasets.
    openml_path: path to openml datasets.
    '''
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-C')
        print('-'*50)
        n, input_dim=max(100000, n_data_to_be_valued+n_val+n_test+1), input_dim
        data = np.random.normal(size=(n,input_dim))
        # beta_true = np.array([2.0, 1.0, 0.0, 0.0, 0.0]).reshape(input_dim,1)
        beta_true = np.random.normal(size=input_dim).reshape(input_dim,1)
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'pol':
        print('-'*50)
        print('pol')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'jannis':
        print('-'*50)
        print('jannis')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/jannis_43977.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'lawschool':
        print('-'*50)
        print('law-school-admission-bianry')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/law-school-admission-bianry_43890.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'fried':
        print('-'*50)
        print('fried')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/fried_901.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'vehicle_sensIT':
        print('-'*50)
        print('vehicle_sensIT')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'electricity':
        print('-'*50)
        print('electricity')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/electricity_44080.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == '2dplanes':
        print('-'*50)
        print('2dplanes_727')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'creditcard':
        print('-'*50)
        print('creditcard')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'covertype':
        print('-'*50)
        print('Covertype')
        print('-'*50)
        from sklearn.datasets import fetch_covtype
        data, target=fetch_covtype(data_home=clf_path, return_X_y=True)
    elif dataset == 'nomao':
        print('-'*50)
        print('nomao')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/nomao_1486.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'webdata_wXa':
        print('-'*50)
        print('webdata_wXa')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/webdata_wXa_350.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
    elif dataset == 'MiniBooNE':
        print('-'*50)
        print('MiniBooNE')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MiniBooNE_43974.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']         
    elif dataset == 'fabert':
        print('-'*50)
        print('fabert')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/fabert_41164.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'gas_drift':
        print('-'*50)
        print('gas_drift')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/gas-drift_1476.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'har':
        print('-'*50)
        print('har')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/har_1478.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'musk':
        print('-'*50)
        print('musk')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/musk_1116.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y']     
    elif dataset == 'magictelescope':
        print('-'*50)
        print('MagicTelescope')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/MagicTelescope_44073.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'ailerons':
        print('-'*50)
        print('ailerons')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/ailerons_734.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'eye_movements':
        print('-'*50)
        print('eye_movements')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/eye_movements_1044.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    elif dataset == 'pendigits':
        print('-'*50)
        print('pendigits')
        print('-'*50)
        data_dict=pickle.load(open(openml_path+'/pendigits_1019.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] 
    else:
        assert False, f"Check {dataset}"

    (X, y), (X_val, y_val), (X_test, y_test) = preprocess_and_split_dataset(data, target,  n_data_to_be_valued, n_val, n_test)   

    return (X, y), (X_val, y_val), (X_test, y_test) 

def preprocess_and_split_dataset(data, target, n_data_to_be_valued, n_val, n_test, is_classification=True):
    if is_classification is True:
        # classification
        target = target.astype(np.int32)
    else:
        # regression
        target_mean, target_std= np.mean(target, 0), np.std(target, 0)
        target = (target - target_mean) / np.clip(target_std, 1e-12, None)
    
    ind=np.random.permutation(len(data))
    data, target=data[ind], target[ind]

    data_mean, data_std= np.mean(data, 0), np.std(data, 0)
    data = (data - data_mean) / np.clip(data_std, 1e-12, None)
    n_total=n_data_to_be_valued + n_val + n_test

    if len(data) >  n_total:
        X=data[:n_data_to_be_valued]
        y=target[:n_data_to_be_valued]
        X_val=data[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
        y_val=target[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
        X_test=data[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]
        y_test=target[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]
    else:
        assert False, f"Original dataset is less than n_data_to_be_valued + n_val + n_test. {len(data)} vs {n_total}. Try again with a smaller number for validation or test."

    print(f'Train X: {X.shape}')
    print(f'Val X: {X_val.shape}') 
    print(f'Test X: {X_test.shape}') 
    print('-'*30)
    
    return (X, y), (X_val, y_val), (X_test, y_test)



