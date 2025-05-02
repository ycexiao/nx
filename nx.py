from train_model import *
from analyze_scores import *


# set load_path
elements = ["Ti", "Fe", "Mn", "Cu"]
file_names = [
    element + "_collection.json" for element in elements
]  # filename or path to the collection
load_dir = "datasets"
load_path = [os.path.join(load_dir, file_names[i]) for i in range(len(file_names))]


# set features of interest
feature_options = ["xanes","x_pdf", "n_pdf", 'nx_pdf', 'diff_x_pdf', 'diff_n_pdf']
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
target = [
    'cn'
]

fea_tar_model = [
    {'features': features[i],
    'target' : target[j],} for i in range(len(features)) for j in range(len(target))
]

for i in range(len(fea_tar_model)):
    if fea_tar_model[i]['target'] == 'bl':
        fea_tar_model[i]['model'] = RandomForestRegressor()
        fea_tar_model[i]['score_method'] = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'))
    else:
        fea_tar_model[i]['model'] = RandomForestClassifier()
        fea_tar_model[i]['score_method'] = make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)/np.mean(y_true))
    
# set training configs
tune_hyper = True
results_path = 'tmp.pickle'

n_estimators = [25, 50, 100, 200, 300]
max_features = [10, 15, 20, 25, 30, 35]
default_model_params = {
    'n_estimators': n_estimators,
    'max_features': max_features
}


total_start = time.time()
total_outs = []
for j in range(len(elements)):
    print("Start on element {}".format(elements[j]))
    for i in range(len(fea_tar_model)):
        round_start = time.time()
        print("Start for \n\tFeatures:{}\n\tTarget:{}".format(fea_tar_model[i]['features'], fea_tar_model[i]['target']))
        dict = {
            'element': elements[j],
            'features': fea_tar_model[i]['features'],
            'target': fea_tar_model[i]['target']
        }
        
        out_dict, find_data = backpropagate(results_path, **dict)
        if find_data:
            print("Use trained parameters. Params: {}".format(out_dict))
            fea_tar_model[i]['model_params'] = out_dict['model_params']
            out = train_model(  # main step
                load_path[j],
                tune_hyper=False,
                **fea_tar_model[i]
            )
        else:
            fea_tar_model[i]['model_params'] = default_model_params
            out = train_model(
                load_path[j],
                tune_hyper=True,
                **fea_tar_model[i]
            )

        out = wrap_results(out, **dict)
        out['element'] = elements[j]
        total_outs.append(out)
        # total_outs.append(out)
        print("{}/{} round finished, coust {} seconds in this round.".format(
            i+j*len(fea_tar_model), len(fea_tar_model)*len(elements), time.time()-round_start)
            )

        if i>1:
            break        
    print('\n\n')
    break

    
# with open('tmp.pickle', 'wb') as f:
#     pickle.dump(total_outs, f)
print("Total {} seconds".format(time.time() - total_start))