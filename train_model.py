import json
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, root_mean_squared_error, make_scorer
import time
import pickle

from analyze_scores import *


def get_model_data(data_path):
    """Get data from datasets and do some basic data filtering.

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
    with open(data_path, "r") as f:
        docs = json.load(f)

    xanes = np.zeros([len(docs), len(docs[0]["xanes"])])
    x_pdf = np.zeros([len(docs), len(docs[0]["x_pdf"])])
    n_pdf = np.zeros([len(docs), len(docs[0]["n_pdf"])])
    diff_x_pdf = np.zeros([len(docs), len(docs[0]["diff_x_pdf"])])
    diff_n_pdf = np.zeros([len(docs), len(docs[0]["diff_n_pdf"])])

    Y = np.zeros([len(docs), 3])

    masks = np.zeros(len(docs), dtype=bool)

    for i in range(len(docs)):
        try:
            xanes[i] = docs[i]["xanes"]
            x_pdf[i] = docs[i]["x_pdf"]
            n_pdf[i] = docs[i]["n_pdf"]
            diff_x_pdf[i] = docs[i]["diff_x_pdf"]
            diff_n_pdf[i] = docs[i]["diff_n_pdf"]
            Y[i, 0] = docs[i]["cs"]
            Y[i, 1] = docs[i]["cn"]
            Y[i, 2] = docs[i]["bl"]
            masks[i] = 1
        except ValueError:
            continue

    X = np.hstack([xanes, x_pdf, n_pdf, diff_x_pdf, diff_n_pdf])
    return X[masks], Y[masks]


def get_model_features(X, names, feature_length=200, force_length=100):
    """Select the features from the loaded dats. Information about the order and
    length of each features are known outside of this function.

    Parameters
    ----------
    X: array_like
        Inputs of the ML model
    names: list.
        str of the features want to use
        options: 'xanes', 'x_pdf', 'n_pdf', 'diff_x_pdf' 'diff_n_pdf'
    feature_length: int
        length of the feature in the datasets.
    force_length: int
        length of the feature we want to use as model input.

    Returns
    -------
    new_X: array_like
    """
    optional_names = ["xanes", "x_pdf", "n_pdf", "diff_x_pdf", "diff_n_pdf", 'nx_pdf']
    dict_names = {optional_names[i] : (i*feature_length, (i+1)*feature_length) for i in range(len(optional_names))}
    new_X = np.zeros([len(X), force_length * len(names)])
    new_X_inds = [(i*force_length, (i+1)*force_length) for i in range(len(names))]

    for i in range(len(names)):
        if names[i] != 'nx_pdf':
            feature = X[:, dict_names[names[i]][0]:dict_names[names[i]][1]]
        else:
            feature = X[:, dict_names['x_pdf'][0]: dict_names['x_pdf'][1]] - \
                X[:, dict_names['n_pdf'][0] : dict_names['n_pdf'][1]]
        for j in range(len(X)):
            new_X[j, new_X_inds[i][0]:new_X_inds[i][1]] = np.interp(
                np.linspace(0, len(feature), force_length),
                np.linspace(0, len(feature), feature_length),
                feature[j],
            )

    return new_X


def get_model_target(Y, name):
    """Select the target used for training. Information about the order and
    length of each target are known outside this function.

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
    optional_names = ["cs", "cn", "bl"]
    dict_names = {optional_names[i]: i for i in range(len(optional_names))}
    new_Y = np.zeros(len(Y))
    new_Y = Y[:, dict_names[name]]
    return new_Y


def train_model_hyper(model, X, y, param_grid, score_method, show=False):
    """Use cv to hyper-tune the model.

    Parameters
    ----------
    model:
        sklearn model to be trained
    X: array_like
    y: array_like
    model_params: dict
        dict of params to be tuning

    """
    grid = GridSearchCV(model, param_grid=param_grid, scoring=score_method)
    grid.fit(X, y)
    if show:
        pass
    return grid.best_params_


def train_model(
    data_path,
    model,
    features,
    target,
    model_params,
    score_method,
    tune_hyper=True,
    n_itr=5,
):
    """
    1. load data, choose feautres and targets
    2. cross_val on trainning set to get param
    3. for each (features, target) combination
        3.1. train the model
        3.2. redo train_test_split and repeat 2.1-2.2 <n_iter> times and store the scores

    Parameters
    ----------
    data_path: str
        path to the data.
    model: 
        sklearn model used to train
    features: list of str
    target: str
    grid_search_params: list
    score_method: 
    dump_prefix: str
        specifying additional information associated with te datasets.
    """
    # set data
    X, Y = get_model_data(data_path)
    mask = np.all(~np.isnan(Y), axis=1)
    X, Y = X[mask], Y[mask]

    X = get_model_features(X, features)
    y = get_model_target(Y, target)

    # start train
    test_scores = []
    train_scores = []
    start = time.time()

    if tune_hyper:        
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model_params = train_model_hyper(
                model, X_train, y_train, param_grid=model_params, score_method=score_method
            )
        print('Tune parameter finished. Cost {} seconds. Params: {}'.format(time.time()-start, model_params))    
    
    for i in range(n_itr):
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model.set_params(**model_params)
        model.fit(X_train, y_train)

        test_scores.append(score_method(estimator=model, X=X_test, y_true=y_test))
        train_scores.append(score_method(estimator=model, X=X_train, y_true=y_train))

        end = time.time()
        print(
            "{} iteration finished. Cost {} seconds. Score {}".format(
                i, end - start, test_scores[-1]
            )
        )
        
    scores = np.array([train_scores, test_scores])
    return model, scores, model_params

def backpropagate(results_path, **dict):
    if not os.path.exists(results_path):
        raise FileNotFoundError
    
    with open(results_path, 'rb') as f:
        trained_results = pickle.load(f)

    conditions = [[generate_condition(key, val) for key, val in dict.items()]]
    data = filter_data(trained_results, *conditions)
    
    if len(data)==0:
        out_dict = {}
        return out_dict, False
    else:
        flag = True
        out_dict = {}
        for key, val in data[0].items():
            out_dict[key] = val
        return out_dict, True

def show_dictionary(dict):
    invalid = []
    for key, val in dict.items():
        if isinstance(val, list):
            print(key+': '+ str([val[i] for i in range(len(val)) if i < 5]))
        else:
            invalid.append(key)
    print('Skip: '+str(invalid))

def wrap_results(out, **dict):
    dict['model'] = out[0]
    dict['train_scores'] = out[1][0]
    dict['test_scores'] = out[1][1]
    dict['model_params'] = out[2]
    return dict


if __name__ == "__main__":
    # set load_path
    elements = ["Ti", "Fe", "Mn", "Cu"]
    file_names = [
        element + "_collection.json" for element in elements
    ]  # filename or path to the collection
    load_dir = "datasets"
    load_path = [os.path.join(load_dir, file_names[i]) for i in range(len(file_names))]

    # set features of interest
    feature_options = ["x_pdf", "n_pdf", 'nx_pdf', 'diff_x_pdf', 'diff_n_pdf']
    one_features = [[f] for f in feature_options]
    two_features = [
        [feature_options[i], feature_options[j]]
        for i in range(len(feature_options))
        for j in range(len(feature_options))
        if (j < i) and (j != i)
    ]
    features = []
    features.extend(one_features)
    features.extend(two_features)

    # set target of interest

    # target_options = ['cn', 'cs', 'bl']
    # target = [[t] for t in target_options]
    target = [
        'cn'
    ]  # model, param_grid and score_method should be adjust for regression.

    cls_param = {
        'model': RandomForestClassifier(),
        'model_params': {
            'n_estimators': [25, 50, 100, 200, 300],
            'max_features': [10, 15, 20, 25, 30, 35] 
        },
        'score_method' : make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')),
    }

    reg_param = {
        'model' : RandomForestRegressor(),
        'model_params' : {
            'n_estimators': [25, 50, 100, 200, 300],
            'max_features': [10, 15, 20, 25, 30, 35] 
        },
        'score_method' : make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)/np.mean(y_true))
    }
    model_params = [cls_param if target[i] in ['cn', 'cs'] else reg_param for i in range(len(target))]

    fea_tar_model = [{
        **{'features': features[i],
        'target' : target[j],},
        **model_params[j]}
        for i in range(len(features)) for j in range(len(target))
    ]


    total_start = time.time()
    total_outs = []
    for j in range(len(elements)):
        print("Start on element {}".format(elements[j]))
        for i in range(len(fea_tar_model)):
            print("Start for \n\tFeatures:{}\n\tTarget:{}".format(fea_tar_model[i]['features'], fea_tar_model[i]['target']))
            dict = {
                'element': elements[j],
                'features': fea_tar_model[i]['features'],
                'target': fea_tar_model[i]['target']
            }

            out = train_model(  # main step
                load_path[j],
                **fea_tar_model[i]
            )
            out = wrap_results(out, **dict)

            total_outs.append(out)
                    
        print('\n\n')
        
    with open('tmp.pickle', 'wb') as f:
        pickle.dump(total_outs, f)
        
    print("Total {} seconds".format(time.time() - total_start))


    # check length
    # from matplotlib import pyplot as plt
    # with open(load_path[0], 'r') as f:
    #     docs = json.load(f)

    # ind = int(np.random.random()*len(docs))
    # fig, ax = plt.subplots(1,4)
    # ax[0].plot(docs[ind]['pdf'])
    # ax[1].plot(docs[ind]['x_pdf'])
    # ax[2].plot(docs[ind]['diff_x_pdf'])
    # ax[3].plot(docs[ind]['diff_n_pdf'])
    # plt.show()
    # print(len(docs[0]['pdf']))
    # print(len(docs[0]['xanes']))

    # check y
    # for i in range(len(load_path)):
    #     X, Y = get_model_data(load_path[i])
    #     mask = np.all(~np.isnan(Y), axis=1)
    #     print(np.all(np.isnan(Y[mask])))
    # print(Y[391])
