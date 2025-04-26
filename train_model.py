import json
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, root_mean_squared_error, make_scorer
import time
import pickle


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
    """Select the features used for training. Information about the order and
    length of each features are known outside this function.

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


def preprocess_data(X,Y):
    pass


def get_model_target(Y, name):
    """Select the targets used for training. Information about the order and
    length of each targets are known outside this function.

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
    grid_search_params,
    score_method,
    dump_prefix,
    dump_dir,
    dump=False,
    n_itr=5,
):
    """
    1. load data, choose feautres and targets
    2. for each (features, target) combination
        2.1. cross_val on trainning set to get param
        2.2. train the model
        2.3. redo train_test_split and repeat 2.1-2.2 <n_iter> times and store the scores
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
    for i in range(n_itr):
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model_params = train_model_hyper(
            model, X_train, y_train, param_grid=grid_search_params, score_method=score_method
        )

        model.set_params(**model_params)
        model.fit(X_train, y_train)

        test_scores.append(score_method(estimator=model, X=X_test, y_true=y_test))
        train_scores.append(score_method(estimator=model, X=X_train, y_true=y_train))

        end = time.time()
        print(
            "{} iteration finished. Cost {} seconds. Param {}".format(
                i, end - start, model_params
            )
        )
        print(test_scores[-1])
    scores = np.array([train_scores, test_scores])

    # set dump
    if dump:
        dump_name = (
            dump_prefix + "-F-" + "-".join(features) + "-T-" + target + ".pickle"
        )
        dump_path = os.path.join(dump_dir, dump_name)
        with open(dump_path, "wb") as f:
            pickle.dump(scores, f)

    return scores


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
        "cs", 'cn'
    ]  # model, param_grid and score_method should be adjust for regression.

    cls_param = {
        'model': RandomForestClassifier(),
        'grid_search_params': {
            'n_estimators': np.arange(40,70,5)
        },
        'score_method' : make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')),
    }

    reg_param = {
        'model' : RandomForestRegressor(),
        'grid_search_params' : {
            'n_estimators' : np.arange(40,70,5)
        },
        'score_method' : make_scorer(root_mean_squared_error)
    }
    model_params = [cls_param if target[i] in ['cn', 'cs'] else reg_param for i in range(len(target))]

    fea_tar_model = [{
        **{'features': features[i],
        'target' : target[j],},
        **model_params[j]}
        for i in range(len(features)) for j in range(len(target))
    ]
        
    # for i in range(len(fea_tar_model)):
    #     print(fea_tar_model[i])
    
    
    print(fea_tar_model[0])
    # dump the params
    # dump_path = 'results/params.pickle'
    # with open(dump_path, 'wb') as f:
    #     pickle.dump(fea_tar_model, f)


    total_start = time.time()

    for j in range(len(elements)):
        print("Start on element {}".format(elements[j]))
        for i in range(len(fea_tar_model)):
            print("  Start for \n\tFeatures:{}\n\tTarget:{}".format(fea_tar_model[i]['features'], fea_tar_model[i]['target']))

            scores = train_model(  # main step
                load_path[j],
                dump_dir="results",
                dump_prefix=elements[j],
                dump=True,
                **fea_tar_model[i]
            )


        print('\n\n')

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
