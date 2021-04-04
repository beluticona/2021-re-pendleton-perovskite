import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def mcc(y_true, y_pred): return matthews_corrcoef(y_true, y_pred)


def sup1(y_true, y_pred): return np.sum(y_true)


def sup0(y_true, y_pred): return len(y_true) - np.sum(y_true)


def columns_to_scale(column_list, std_dict, norm_dict):
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
