import numpy as np
import random
from sklearn.model_selection import train_test_split
from configs import DataConfig, model_config, model_lookup

def concat_source_data(X, toxic_data, normal_data, domain, layer, random_seed):
    np.random.seed = random_seed
    toxic_sources = [source for source in toxic_data if source in X[domain]]
    # filter out anything that is None in the X dict
    toxic_sources = [source for source in toxic_sources if X[domain][source][layer] is not None]
    normal_sources = [source for source in normal_data if source in X[domain]]
    normal_sources = [source for source in normal_sources if X[domain][source][layer] is not None]
    X_toxic = np.concatenate([X[domain][source][layer] for source in toxic_sources])
    X_normal = np.concatenate([X[domain][source][layer] for source in normal_sources])
    
    if X_toxic.shape[0] < X_normal.shape[0]:
        sample_idx = np.random.choice(X_normal.shape[0], size=X_toxic.shape[0], replace=False)
        X_normal = X_normal[sample_idx,:]
    else:
        sample_idx = np.random.choice(X_toxic.shape[0], size=X_normal.shape[0], replace=False)
        X_toxic = X_toxic[sample_idx,:]

    y_toxic = np.zeros(X_toxic.shape[0])
    y_normal = np.ones(X_normal.shape[0])

    X = np.concatenate([X_toxic, X_normal])
    y = np.concatenate([y_toxic, y_normal])

    return X, y


def get_single_domain_data(X_dict, data_config, domain, layer, random_seed):
    np.random.seed = random_seed
    toxic_data = data_config.toxic_data
    normal_data = data_config.normal_data

    X, y = concat_source_data(X_dict, toxic_data, normal_data, domain, layer, random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=model_config.seed)
    
    return X_train, X_test, y_train, y_test


def get_hold_one_out_data(X_dict, data_config, test_domain, layer, random_seed):
    np.random.seed = random_seed
    toxic_data = data_config.toxic_data
    normal_data = data_config.normal_data
    X_train, y_train = [], []
    
    for domain in data_config.domains:
        print(domain)
        if domain!=test_domain:
            X, y = concat_source_data(X_dict, toxic_data, normal_data, domain, layer, random_seed)
            X_train.append(X)
            y_train.append(y)
        else:
            X_test, y_test = concat_source_data(X_dict, toxic_data, normal_data, domain, layer, random_seed)
        print(len(X_train))
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    return X_train, X_test, y_train, y_test

def get_mixed_data(X_dict, data_config, layer, random_seed):
    np.random.seed = random_seed
    toxic_data = data_config.toxic_data
    normal_data = data_config.normal_data
    X_list, y_list = [], []
    
    for domain in data_config.domains:
        X, y = concat_source_data(X_dict, toxic_data, normal_data, domain, layer, random_seed)
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_seed)

    return X_train, X_test, y_train, y_test




                


