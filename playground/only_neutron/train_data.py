
## load data
import json
import numpy as np 

dump_file = 'playground/only_neutron/neutron_collection.json'
with open(dump_file, 'r') as file:
    example_collection = json.load(file)

print(example_collection[0].keys())  # Print the first entry to check the structure

y_cn = [example_collection[i]['cn'] for i in range(len(example_collection))]
y_cs = [example_collection[i]['cs'] for i in range(len(example_collection))]
y_bl = [example_collection[i]['bl'] for i in range(len(example_collection))] 
X = [example_collection[i]['npdf'][1] for i in range(len(example_collection))]

y_cn = np.array(y_cn)
y_cs = np.array(y_cs)
y_bl = np.array(y_bl)
X = np.array(X)

print("Shape of X:", X.shape)
print("Shape of y:", y_cn.shape)


## train clf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def quick_rf_clf_train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    n_estimators = [25, 50, 100, 200, 300]
    max_features = [6, 8, 10, 15, 20, 25, 30, 35]
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features
    }
    clf = GridSearchCV(RandomForestClassifier(max_features='sqrt'), param_grid, 
                    scoring='f1_weighted',  # Use weighted F1 score to handle class imbalance
                    cv=5, 
                    verbose=1,
                    n_jobs=-1) 

    clf.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(clf.best_score_))
    print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
    print( "Best parameters found: ", clf.best_params_)


# quick_rf_clf_train(X, y_cn)
# quick_rf_clf_train(X, y_cs)  


## train rg
# from sklearn.ensemble import RandomForestRegressor
# X = X[~np.isnan(y_bl)]
# y_bl = y_bl[~np.isnan(y_bl)]
# X_train, X_test, y_train, y_test = train_test_split(X, y_bl, test_size=0.2, random_state=42)


# # print(np.isnan(X_train).any(), np.isnan(y_train).any())
# arr_n_estimators = [25, 50, 100, 200, 300]
# arr_max_features = [10, 15, 20, 25, 30] 
# param_grid_rg = {
#     'n_estimators': arr_n_estimators,
#     'max_features': arr_max_features
# }
# rg = GridSearchCV(RandomForestRegressor(), param_grid_rg,
#                     scoring='r2',  # Use negative MSE for regression
#                     cv=5, 
#                     verbose=1,
#                     n_jobs=-1)
# rg.fit(X_train, y_train)
# print("Best cross-validation score (MSE): {:.2f}".format(rg.best_score_))
# print("Test set score (MSE): {:.2f}".format(rg.score(X_test, y_test)))