import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold

'''
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
'''
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
import numpy as np


def get_sol_ud_model_columns(df_columns):
    sol_ud_model = list(filter(lambda column_name: column_name.startswith('_rxn_') or column_name.startswith(
        '_feat_') and not column_name.startswith('_rxn_v0'), df_columns))
    sol_ud_model.remove('_rxn_organic-inchikey')
    return sol_ud_model


# Select columns names involved in each model
def filter_data_for_sol_v(df):
    sol_ud_model = get_sol_ud_model_columns(df.columns.to_list())

    sol_v_model = list(filter(lambda column_name: not column_name.startswith('_rxn_M_'), sol_ud_model))

    # Select data involved in each model
    sol_v_data = df[sol_v_model].reset_index(drop=True)

    return sol_v_data


def filter_data_for_sol_ud(df):
    sol_ud_model = get_sol_ud_model_columns(df.columns.to_list())

    sol_ud_data = df[sol_ud_model].reset_index(drop=True)

    return sol_ud_data


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def mcc(y_true, y_pred): return matthews_corrcoef(y_true, y_pred)
def sup1(y_true, y_pred): return np.sum(y_true)
def sup0(y_true, y_pred): return len(y_true) - np.sum(y_true)


def std_train_test(data, model_parameters, crystal_score, dataset_name, results):
    x_train, x_test, y_train, y_test = train_test_split(data, crystal_score, test_size=0.2, random_state=42)

    clf = KNeighborsClassifier(leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=8, p=20)
    param_grid = {'weights': ['uniform', 'distance'],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                  'n_neighbors': range(3, 9, 2)
                  }

    clf_dict = {'estimator': clf,
                'opt': model_parameters['hyperparam_opt'],
                'param_grid': param_grid
                }

    cv = model_parameters['cv']

    if cv <= 1:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

        for metric in results:
            value = {
                'dataset_index': dataset_name,
                'cv': 1,
                # 'matrix':confusion_matrix(y_test, pred),
                'precision_positive': precision[1],
                'recall_positive': recall[1],
                'f1_positive': f1[1],
                'support_negative': support[0],
                'support_positive': support[1],
                'matthewCoef': matthews_corrcoef(y_test, y_pred)
            }[metric]
            results[metric].append(value)

    else:
        # metrics to track
        scoring = {  # 'tp': make_scorer(tp),
            'precision': 'precision',
            'recall': 'recall',
            'mcc': make_scorer(mcc),
            'support_negative': make_scorer(sup0),
            'support_positive': make_scorer(sup1),
            'f1': 'f1'}
        # 'tn': make_scorer(tn),
        # 'fp': make_scorer(fp),
        # 'fn': make_scorer(fn),

        # shuffle batched experimental data into discrete experiments
        scores = cross_validate(clf, data, crystal_score,
                                cv=KFold(model_parameters['cv'], shuffle=True, random_state=1),
                                scoring=scoring,
                                return_train_score=True,
                                return_estimator=True)
        for i in range(cv):
            for metric in results:
                value = {
                    'dataset_index': dataset_name,
                    'cv': i,
                    'precision_positive': scores['test_precision'][i],
                    'recall_positive': scores['test_recall'][i],
                    'f1_positive': scores['test_f1'][i],
                    'support_negative': scores['test_support_negative'][i],
                    'support_positive': scores['test_support_positive'][i],
                    'matthewCoef': scores['test_mcc'][i]
                }[metric]
                results[metric].append(value)

    '''
    scores = cross_validate(clf, data, crystal_score,
                            cv=KFold(cv, shuffle=True),  #shuffle batched experimental data into descrete experiments
                            scoring=scoring, 
                            return_train_score=True,
                            return_estimator=True)
    return scores, clf

    clf_pipe = Pipeline(steps=[('transform', None), ('clf', model)])

    # @TODO:add if hyperparam_opt ON or OFF
    # default ON
    clf = GridSearchCV(clf_pipe, param_grid=param_grid, refit=True, cv=5, n_jobs=8)
    clf.fit(X, y)
    clf = clf.best_estimator_

    pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, pred)
    matt_coeff = matthews_corrcoef(y_test, pred)
    return {'pred':pred,
            'cm': cm,
            'precision':precision,
            'recall':recall,
            'f1':f1,
            'matt_coeff': matt_coeff
            }

    '''
