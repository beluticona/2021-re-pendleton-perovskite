from sklearn.preprocessing import Normalizer
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def mcc(y_true, y_pred): return matthews_corrcoef(y_true, y_pred)


def sup1(y_true, y_pred): return np.sum(y_true)


def sup0(y_true, y_pred): return len(y_true) - np.sum(y_true)


def make_classifier(model_parameters):
    if model_parameters['method'] == 1:
        clf = KNeighborsClassifier(leaf_size=30, metric='minkowski',
                                   metric_params=None, n_jobs=8, p=20)
        param_grid = {'weights': ['uniform', 'distance'],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'n_neighbors': range(3, 9, 2)
                      }
    elif model_parameters['method'] == 2:
        gbt = GradientBoostingClassifier(random_state=42)
        param_grid = {'min_samples_split': range(2, 10, 2),
                      'min_samples_leaf': range(2, 5),
                      'max_depth': range(2, 7),
                      'learning_rate': [0.05, 0.10, 0.15, 0.20]
                      }

    clf_dict = {'estimator': clf,
                'opt': model_parameters['hyperparam_opt'],
                'param_grid': param_grid
                }
    return clf, clf_dict


def std_norm_cols(column_list, std_dict, norm_dict):
    curated_list = []
    for header_prefix in std_dict:
        if std_dict[header_prefix] == 1:
            for column in column_list:
                if header_prefix in column:
                    curated_list.append(column)
    for header_prefix in norm_dict:
        if norm_dict[header_prefix] == 1:
            for column in column_list:
                if header_prefix in column:
                    curated_list.append(column)
    return curated_list


def feat_scaling(parameters, data_columns):
    requested_norm = [dataset_name for (dataset_name, required) in parameters["norm"].items() if required]
    requested_sdt = [dataset_name for (dataset_name, required) in parameters["std"].items() if required]

    if len(requested_norm) + len(requested_sdt) == 0:
        return None
    else:
        curated_columns = std_norm_cols(data_columns, parameters['std'], parameters['norm'])
        if len(requested_norm) > 0: fun = Normalizer()
        else: fun = StandardScaler()

    return make_column_transformer((fun, curated_columns), remainder='passthrough')


def std_train_test(data, model_parameters, crystal_score, dataset_name, results):
    X, X_test, y, y_test = train_test_split(data, crystal_score, test_size=0.2, random_state=42)
    clf, clf_dict = make_classifier(model_parameters)

    data_preprocess = feat_scaling(model_parameters, data.columns.to_list())
    pipeline = Pipeline([
        ('scale', data_preprocess),
        ('clf', clf)
    ])

    cv = model_parameters['cv']

    if cv <= 1:
        clf.fit(X, y)
        y_pred = clf.predict(X_test)

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

        result_by_metric = {
            'dataset_index': dataset_name,
            'cv': 1,
            # 'matrix':confusion_matrix(y_test, pred),
            'precision_positive': precision[1],
            'recall_positive': recall[1],
            'f1_positive': f1[1],
            'support_negative': support[0],
            'support_positive': support[1],
            'matthewCoef': matthews_corrcoef(y_test, y_pred)
        }

        for metric in results:
            results[metric].append(result_by_metric[metric])

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
        scores = cross_validate(pipeline, data, crystal_score,
                                cv=KFold(cv, shuffle=True, random_state=2),
                                scoring=scoring,
                                return_train_score=True,
                                return_estimator=True)

        metrics_by_name = {
            'precision_positive': scores['test_precision'],
            'recall_positive': scores['test_recall'],
            'f1_positive': scores['test_f1'],
            'support_negative': scores['test_support_negative'],
            'support_positive': scores['test_support_positive'],
            'matthewCoef': scores['test_mcc']
        }

        metrics = results.keys() - {'dataset_index', 'cv'}
        for i in range(cv):
            results['dataset_index'].append(dataset_name)
            results['cv'].append(i)
            for metric in metrics:
                results[metric].append(metrics_by_name[metric][i])

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
