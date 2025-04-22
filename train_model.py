import json
import os 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time
import pickle

def get_model_data(data_path):
    """
    Get data from datasets and do some basic data filtering

    Parameters
    ----------
    data_path: str
        path to load the data

    Returns
    -------
    X: array_like
        inputs of the ML model
    Y: array_like
        outputs of the ML model
    """
    with open(data_path, 'r') as f:
        docs = json.load(f)
    
    xanes = np.zeros([len(docs), len(docs[0]['xanes'])])
    x_pdf = np.zeros([len(docs), len(docs[0]['x_pdf'])])
    n_pdf = np.zeros([len(docs), len(docs[0]['n_pdf'])])
    diff_x_pdf = np.zeros([len(docs), len(docs[0]['diff_x_pdf'])])
    diff_n_pdf = np.zeros([len(docs), len(docs[0]['diff_n_pdf'])])
    
    Y = np.zeros([len(docs), 3])

    masks = np.zeros(len(docs), dtype=bool)

    for i in range(len(docs)):
        try:
            xanes[i] = docs[i]['xanes']
            x_pdf[i] = docs[i]['x_pdf']
            n_pdf[i] = docs[i]['n_pdf']
            diff_x_pdf[i] = docs[i]['diff_x_pdf']
            diff_n_pdf[i] = docs[i]['diff_n_pdf']
            Y[i,0] = docs[i]['cs']
            Y[i,1] = docs[i]['cn']
            Y[i,2] = docs[i]['bl']
            masks[i] = 1
        except ValueError:
            continue
    
    X = np.hstack([xanes, x_pdf, n_pdf, diff_x_pdf, diff_n_pdf])
    return X[masks], Y[masks]


def get_model_features(X, names):
    """
    Select the features used for training.
    Information about the order and length of each features are known outside this function.

    Parameters
    ----------
    X: array_like
        Inputs of the ML model
    names: list. 
        str of the features want to use
        options: 'xanes', 'x_pdf', 'n_pdf', 'diff_x_pdf' 'diff_n_pdf'

    Returns
    -------
    new_X: array_like
    """
    optional_names = ['xanes', 'x_pdf', 'n_pdf', 'diff_x_pdf', 'diff_n_pdf']
    dict_names = {optional_names[i]:i for i in range(len(optional_names))}
    feature_length = 200
    new_X = np.zeros([len(X), feature_length*len(names)])

    
    for i in range(len(names)):
        new_X_start_ind, new_X_end_ind = i*200, (i+1)*200
        X_start_ind, X_end_ind = dict_names[names[i]]*200, (dict_names[names[i]]+1)*200
        new_X[:, new_X_start_ind:new_X_end_ind] = X[:, X_start_ind:X_end_ind]

    return new_X
        
        

def get_model_targets(Y, name):
    """
    Select the targets used for training.
    Information about the order and length of each targets are known outside this function.

    Parameters
    ----------
    X: array_like
        Inputs of the ML model
    name: str 
        str of the target
        options: 'cs', 'cn' or 'bl'
    
    Returns
    -------
    new_Y
    """
    optional_names = ['cs', 'cn', 'bl']
    dict_names = {optional_names[i]:i for i in range(len(optional_names))}
    new_Y = np.zeros(len(Y))
    new_Y = Y[:,dict_names[name]]
    return new_Y
    

def train_model_hyper(model, X, y, param_grid, show=False):
    """
    Use cv to hyper-tune the model.

    Parameters
    ----------
    model: 
        sklearn model to be trained
    X: array_like
    y: array_like
    model_params: dict
        dict of params to be tuning
    """
    grid = GridSearchCV(model, param_grid=param_grid, scoring='f1_weighted')
    grid.fit(X, y)
    if show:
        pass
    return grid.best_params_
    


def train_model(data_path, model, features, target, param_grid, dump_prefix, dump_dir, dump=False, n_itr=5, score_method=f1_score):
    """
    Construct the main function.
    data, feature and targets can vary depending on the function used.
    """
    X, Y = get_model_data(data_path)
    X = get_model_features(X, features)
    y = get_model_targets(Y, target)

    test_scores = []
    train_scores = []
    for i in range(n_itr):
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X,y)

        model_params = train_model_hyper(model, X_train, y_train, param_grid=param_grid)
        
        model.set_params(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_scores.append(score_method(y_test, y_pred, average='weighted'))
        train_scores.append(score_method(model.predict(X_train), y_train, average='weighted'))

        end = time.time()
        print('{} iteration finished. Cost {} seconds. Param {}'.format(i, end-start, model_params))
    
    scores = np.array([train_scores, test_scores])
    if dump:
        dump_name = dump_prefix+'-F-'+'-'.join(features)+'-T-'+target + '.pickle'
        dump_path = os.path.join(dump_dir, dump_name)
        with open(dump_path, 'wb') as f:
            pickle.dump(scores, f)
        
    return scores


if __name__ == '__main__':
    elements = ['Ti', "Fe", "Mn", 'Cu']
    file_names = [element + '_collection.json' for element in elements] # filename or path to the collection
    load_dir = 'datasets'
    load_path = [os.path.join(load_dir, file_names[i]) for i in range(len(file_names))]


    feature_options = ['xanes', 'x_pdf', 'n_pdf', 'diff_x_pdf', 'diff_n_pdf']
    one_features = [[f] for f in feature_options]
    two_features = [[feature_options[i], feature_options[j]] for i in range(len(feature_options)) for j in range(len(feature_options)) if (j < i) and (j!=i)]
    features = []
    features.extend(one_features)
    features.extend(two_features)

    # target_options = ['cn', 'cs', 'bl']
    # target = [[t] for t in target_options]

    target = ['cn', 'cs']  # model, param_grid and score_method should be adjust for regression.

    ft = [[f, t] for f in features for t in target]

    for j in range(len(elements)):
        print("Start on element {}".format(elements[j]))
        for i in range(len(ft)):
            print('  Start for \n\tFeatures:{}\n\tTarget:{}'.format(ft[i][0], ft[i][1]))
            model = RandomForestClassifier()
            param_grid = {
                    'n_estimators' : np.arange(40, 70, 5),
                    }
            scores = train_model(load_path[j], model, features= ft[i][0], target=ft[i][1], param_grid=param_grid,
                dump_dir='results', dump_prefix=elements[j], dump=True)
        print('\n\n')


    


