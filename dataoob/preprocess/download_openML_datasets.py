import os, pickle, argparse
import utils
import openml
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./tmp', type=str, help="A path to save preprocessed datasets")
    parser.add_argument('--use_sample', action='store_true', default=False)
    parser.add_argument('--n_minimum', default=5000, type=int)
    parser.add_argument('--n_maximum', default=100000, type=int)
    parser.add_argument('--p_maximum', default=1000, type=int)
    args = parser.parse_args() 

    DATA_PATH=args.path 
    openml.config.set_cache_directory(DATA_PATH)
    if args.use_sample is True:
        import pandas as pd
        print('Read dataset information from sample_openml.csv')
        df_openml_datasets = pd.read_csv(f'{DATA_PATH}/sample_openml.csv')
    else:
        warnings.warn('It would take a bit of time and memory. You may want to check --use_sample')
        df_openml_datasets = utils.create_df_openml_datasets(n_minimum=args.n_minimum,
                                                               n_maximum=args.n_maximum, 
                                                               p_maximum=args.p_maximum)

    # make necessary directories
    if not os.path.exists(f'{DATA_PATH}/dataset_clf_openml'):
        print(f'In {DATA_PATH}, dataset_clf_openml does not exist.')
        os.makedirs(f'{DATA_PATH}/dataset_clf_openml')
            
    # make necessary directories
    for index, row in df_openml_datasets.iterrows():
        dataset_num, name, task_type = row['dataset_id'], row['name'], row['task_type']
        file_name = name+'_'+str(dataset_num)+'.pkl'

        if not os.path.exists(f'{DATA_PATH}/{file_name}'):
            print(f'{file_name} does not exist. Will download it from openml', flush=True)
            save_dict, file_name, indication, dataset_name = utils.dataset_maker(dataset_num, name, task_type) 
            if indication == 'success':
                if not os.path.exists(f'{DATA_PATH}/dataset_{task_type}_openml/{file_name}'):
                    with open(f'{DATA_PATH}/dataset_{task_type}_openml/{file_name}', 'wb') as handle:
                        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                n_samples=save_dict['n_samples']
                n_variables=len(save_dict['num_names'])+len(save_dict['cat_names'])
                print(f'ID: {dataset_num:<5}, N: {n_samples:<5}, p: {n_variables:<5}, Name: {dataset_name:<30}')
            else:
                print('Indication is not success', dataset_num, dataset_name, indication)
            del save_dict, file_name, indication






