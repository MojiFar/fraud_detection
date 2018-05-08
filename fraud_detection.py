# -*- coding: utf-8 -*-
"""
Created on Thu May 03 16:35:50 2018

@author: 51648
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score,classification_report
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from time import time
from sklearn.svm import SVC


def HyperParam(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
    #smote = SMOTETomek(random_state = 0)
    smote = SMOTE(random_state = 0)
    pipeline = Pipeline([
               ('smote', smote),
               ('clf', SVC(kernel = 'linear')),
                ])

    parameters = {
              'clf__class_weight': (None, 'balanced'),
             }

    gs_clf = GridSearchCV(estimator=pipeline, param_grid=parameters, 
                          n_jobs=-1, verbose=1, cv = 5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    gs_clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % gs_clf.best_score_)
    print("Best parameters set:")
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        #### Cross-validation to check the test set error rate
    scores = cross_val_score(gs_clf, X_train, y_train, scoring='neg_mean_absolute_error')
    print(scores)
    
    
    ## Predicting the test set results
    y_pred = gs_clf.predict(X_test)
    
    print("The accuracy score of {0}".format(accuracy_score(y_test, y_pred)))
    print ('Confusion Matrix:')
    print (confusion_matrix(y_test, y_pred))
    print (accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))
    return y_test, y_pred, X_train

dataset = pd.read_csv('creditcard.csv')
dataset['normAmount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
datasetNew = dataset.drop(['Time','Amount'],axis=1)

X = datasetNew.drop(['Class'], axis = 1)
y = datasetNew[['Class']].copy()
y_ov_test, y_ov_pred, X_train = HyperParam(X,y)





