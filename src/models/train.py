import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

'''
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
'''
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from src.models import utils
from src import constants


def make_classifier(model_parameters):
    if model_parameters['method'] == constants.KNN:
        clf = KNeighborsClassifier(leaf_size=30, metric='minkowski',
                                   metric_params=None, n_jobs=8, p=20)
        param_grid = {'weights': ['uniform', 'distance'],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'n_neighbors': range(3, 9, 2)
                      }
    elif model_parameters['method'] == constants.GBC:
        clf = GradientBoostingClassifier(random_state=42)
        param_grid = {'min_samples_split': range(2, 10, 2),
                      'min_samples_leaf': range(2, 5),
                      'max_depth': range(2, 7),
                      'learning_rate': [0.05, 0.10, 0.15, 0.20]
                      }
    else:
        clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=10000,
                        multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
                        verbose=0)
        param_grid = {'tol': [0.001, 0.0001, 0.00001],
                      'dual': [True, False],
                      'fit_intercept': [True, False],
                      'loss': ['hinge', 'squared_hinge'],
                      'penalty': ['l1', 'l2']
                      }

    clf_dict = {'estimator': clf,
                'opt': model_parameters['hyperparam_opt'],
                'param_grid': param_grid
                }
    return clf_dict


def simple_fit_predict(X, X_test, pipeline, dataset_name, results, y, y_test):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    result_by_metric = {
        'dataset_index': dataset_name,
        'cv': 1,
        'precision_positive': precision[1],
        'recall_positive': recall[1],
        'f1_positive': f1[1],
        'support_negative': support[0],
        'support_positive': support[1],
        'matthewCoef': matthews_corrcoef(y_test, y_pred)
    }
    for metric in result_by_metric.keys():
        results[metric].append(result_by_metric[metric])


def cross_validate_fit_predict(scores, model_parameters, data_columns, results, dataset_name, curated_columns):
    metrics_by_name = {
        'precision_positive': scores['test_precision'],
        'recall_positive': scores['test_recall'],
        'f1_positive': scores['test_f1'],
        'support_negative': scores['test_support_negative'],
        'support_positive': scores['test_support_positive'],
        'matthewCoef': scores['test_mcc']
    }

    folds = model_parameters['cv']
    for i in range(folds):
        results['dataset_index'].append(dataset_name)
        results['cv'].append(i)
        for metric in metrics_by_name.keys():
            results[metric].append(metrics_by_name[metric][i])

    if model_parameters['method'] == constants.GBC:
        features_importance_per_fold = [scores['estimator'][f][1].feature_importances_ for f in range(folds)]
        # reorder column headers from pipeline operations (report correctly!)

        hold_curated = prepare_features_to_be_sort_by_importance(curated_columns, data_columns, results)

        # sort descending [::-1]
        for k in range(folds):
            results[constants.FEAT_VALUES_IMPORTANCE].append(
                features_importance_per_fold[k][np.argsort(features_importance_per_fold[k])[::-1]])
            results[constants.FEAT_NAMES_IMPORTANCE].append(
                hold_curated[np.argsort(features_importance_per_fold[k])[::-1]])


def prepare_features_to_be_sort_by_importance(curated_columns, data_columns, results):
    temp_headers = [col for col in data_columns if col not in curated_columns]
    # if no columns are selected for the pipeline, no columns will be moved
    hold_curated = list(curated_columns)
    hold_curated.extend(temp_headers)
    hold_curated = np.array(hold_curated)
    if constants.FEAT_NAMES_IMPORTANCE not in results:
        results[constants.FEAT_NAMES_IMPORTANCE] = []
        results[constants.FEAT_VALUES_IMPORTANCE] = []
    return hold_curated


def execute_cross_validation(crystal_score, data, folds, pipeline):
    scoring = {  # 'tp': make_scorer(tp),
        'precision': 'precision',
        'recall': 'recall',
        'mcc': make_scorer(utils.mcc),
        'support_negative': make_scorer(utils.sup0),
        'support_positive': make_scorer(utils.sup1),
        'f1': 'f1'}
    # shuffle batched experimental data into discrete experiments
    scores = cross_validate(pipeline, data, crystal_score,
                            cv=KFold(folds, shuffle=True, random_state=2),
                            scoring=scoring,
                            return_train_score=True,
                            return_estimator=True)
    return scores


def std_train_test(data, model_parameters, crystal_score, dataset_name, results):
    X, X_test, y, y_test = train_test_split(data, crystal_score, test_size=0.2, random_state=42)
    data_preprocess, curated_columns = utils.feat_scaling(model_parameters, data.columns.to_list())

    clf_dict = make_classifier(model_parameters)
    clf = clf_dict['estimator']
    pipeline = Pipeline([
        ('scale', data_preprocess),
        ('clf', clf)
    ])

    if model_parameters['cv'] <= 1:
        simple_fit_predict(X, X_test, pipeline, dataset_name, results, y, y_test)
        if model_parameters['method'] == constants.GBC:
            features_importances = pipeline['model'].feature_importances_
            hold_curated = prepare_features_to_be_sort_by_importance(curated_columns, data.columns.to_list(), results)
            results[constants.FEAT_VALUES_IMPORTANCE].append(
                features_importances[np.argsort(features_importances)[::-1]])
            results[constants.FEAT_NAMES_IMPORTANCE].append(hold_curated[np.argsort(features_importances)[::-1]])

    else:
        # metrics to track
        scores = execute_cross_validation(crystal_score, data, model_parameters['cv'], pipeline)
        cross_validate_fit_predict(scores, model_parameters, data.columns.to_list(), results, dataset_name,
                                   curated_columns)


def amine_slip(selected_data, model_parameters, crystal_score, inchis):
    if model_parameters['strat']:
        indicies = {}
        for i, inchi in enumerate(inchis):
            indicies[inchi] = np.array(range(96)) + i*96

        for amine in inchis.keys():
            is_amine = indicies[amine]
            train = list(set(range(selected_data.shape[0])) - set(indicies[amine]))
            X_test_0 = selected_data.iloc[is_amine]
            X_train_0 = selected_data.iloc[train]

            y_test = crystal_score.iloc[is_amine].values.squeeze()
            y_train = crystal_score.iloc[train].values.squeeze()

            yield X_train_0, X_test_0, y_train, y_test
    else:
        for amine in inchis.unique():
            is_amine = (inchis == amine).values
            X_test_0 = selected_data[is_amine]
            X_train_0 = selected_data[~is_amine]

            y_test = crystal_score[is_amine]
            y_train = crystal_score[~is_amine]

            yield X_train_0, X_test_0, y_train, y_test


def get_success_scores(results):
    for col in ['precision', 'recall', 'f1']:
        results[col + '_success'] = results[col].apply(lambda x: x[1])

    results['matthewCoef_success'] = results['matthewCoef']
    return results


def leave_one_out_train_test(data, model_parameters, crystal_score, dataset_name, inchis, results):
    """
    Strat = 1 : Uniformly sampling 96 experiments for each amine
    Strat = 0 : Samples built from all experiments for each amine
    """
    mycv_return = amine_slip(data, model_parameters, crystal_score, inchis)

    for fold in range(model_parameters['cv']):
        for inchi, (X_train, X_test, y_train, y_test) in zip(inchis.unique(), mycv_return):
            # for each dataset (model0, model1 concentrations)
            clf_dict = make_classifier(model_parameters)
            clf = clf_dict['estimator']
            data_preprocess, curated_columns = utils.feat_scaling(model_parameters, data.columns.to_list())
            pipeline = Pipeline([
                ('scale', data_preprocess),
                ('clf', clf)
            ])
            simple_fit_predict(X_train, X_test, pipeline, dataset_name, results, y_train, y_test)
            utils.record_amine_info(inchi, results)
            if model_parameters['method'] == constants.GBC:
                features_importances = pipeline['model'].feature_importances_
                hold_curated = prepare_features_to_be_sort_by_importance(curated_columns, data.columns.to_list(), results)
                results[constants.FEAT_VALUES_IMPORTANCE].append(
                    features_importances[np.argsort(features_importances)[::-1]])
                results[constants.FEAT_NAMES_IMPORTANCE].append(hold_curated[np.argsort(features_importances)[::-1]])

            utils.translate_inchi_key(inchi, results)
