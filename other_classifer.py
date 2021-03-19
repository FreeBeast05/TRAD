from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def other_classiffiers(data):
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=33, shuffle=True)

    x_test = test_df.copy()
    x_test.drop('price', axis='columns', inplace=True)

    x_train = train_df.copy()
    x_train.drop('price', axis='columns', inplace=True)

    y_train = train_df['price']
    y_test = test_df['price']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    print('#            Naive Bayes        #')
    from sklearn.naive_bayes import GaussianNB
    GNBclassifier = GaussianNB()
    GNBclassifier.fit(X_train, y_train)
    y_pred1 = GNBclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred1))

    print('#            K Neighbors        #')
    from sklearn.neighbors import KNeighborsClassifier
    KNclassifier = KNeighborsClassifier(n_neighbors=5)
    KNclassifier.fit(X_train, y_train)
    y_pred2 = KNclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred2))

    print('#            Decision Tree        #')
    from sklearn.tree import DecisionTreeClassifier
    DTclassifier = DecisionTreeClassifier()
    DTclassifier.fit(X_train, y_train)
    y_pred3 = DTclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred3))

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
    TPR = float(TP) / (TP + FN)
    TNR = float(TN) / (TN + FP)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print('True Positive: {}'.format(TP))
    print('True Negative: {} '.format(TN))
    print('False Positive: {}'.format(FP))
    print('False Negative: {}'.format(FN))
    print('True Positive Rate: {}'.format(TPR))
    print('True Negative Rate: {}'.format(TNR))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(rec))
df=pd.read_csv('file_df_all')
other_classiffiers(df)