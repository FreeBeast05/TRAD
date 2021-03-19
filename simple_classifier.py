import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def print_all_accuracy_metris(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: {}\nROC AUC: {}".format(acc, roc_auc))
    y_test = np.array(y_test)
    y_pred=np.array(y_pred)
    TP = np.sum(y_test * y_pred)
    TN = np.sum(y_test + y_pred == 0)
    FP = np.sum((y_test == 0) * (y_pred == 1))
    FN = np.sum((y_test == 1) * (y_pred == 0))
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print('True Positive: {}'.format(TP))
    print('True Negative: {} '.format(TN))
    print('False Positive: {}'.format(FP))
    print('False Negative: {}'.format(FN))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(rec))

def read_data():
    input_file = open('file_df_1', 'r')
    data = {}
    labels = {}
    key_obj = 0
    for row in input_file:
        str_list = row.strip('\n').split(',')
        labels[key_obj] = int(str_list[-1])
        data[key_obj] = [int(i) for i in str_list[0:-1]]
        key_obj += 1
    input_file.close()
    return data, labels
def simple_classifer(train_positive,train_negative, test_data):
    pos=train_positive.values()
    neg=train_negative.values()
    test=test_data.values()
    predictions = []
    for i in test:
        positive = 0
        negative = 0
        res=0
        for j in pos:
            for k, l in zip(i, j):
                if k==l:
                    res+=1
            positive += res / len(j)
        positive = positive / len(pos)
        res=0
        for j in neg:
            for k, l in zip(i, j):
                if k == l:
                    res += 1
            negative += res / len(j)
        negative = negative / len(neg)
        if (positive < negative):
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions


def classify_test_data(train_data, test_data, labels):
    label_positive = 1
    label_negative = 0
    sample = {label_positive: [], label_negative: []} # 1(позитив) или 0 (негатив) в целевом признаке
    for i, l in labels.items():
        if l == label_positive:
            sample[label_positive].append(i)
        elif l == label_negative:
            sample[label_negative].append(i)
        else:
            print('Label error', l, i)
    train_positive = {}
    train_negative = {}
    for obj in sample[label_positive]:
        train_positive[obj] = train_data[obj]
    for obj in sample[label_negative]:
        train_negative[obj] = train_data[obj]
    predictions=simple_classifer(train_positive,train_negative, test_data)
    return predictions


data, labers= read_data()

train_portion = 0.8
train_ind = set(random.sample(range(len(data)), round(train_portion*len(data))))
test_ind = set(range(len(data))) - train_ind

train_data = {}
train_labels = {}
for ind in train_ind:
    train_data[ind] = data[ind]
    train_labels[ind] = labers[ind]

test_data = {}
test_labers=[]
for ind in test_ind:
    test_data[ind] = data[ind]
    test_labers.append(labers[ind])
y_pred = classify_test_data(train_data, test_data, train_labels)
print_all_accuracy_metris(test_labers, y_pred)

