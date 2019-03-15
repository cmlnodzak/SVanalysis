#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:58:27 2019

@author: cmlnodzak
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegressionCV
import scipy, importlib, pprint, warnings
import glmnet_python
from glmnet import glmnetSet
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetPlot import cvglmnetPlot
from glmnetPrint import glmnetPrint
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict

import sys
sys.stdout = sys.__stdout__ 
log_file = open("Leukemia.models.log.txt","w")

sys.stdout = log_file






# get expression dataset for all genes and samples
# make a list of the subtypes 
groups = ["BCR-ABL1", "CRLF2", "E2A_PBX1","ERG","ETV6_RUNX1","Aneuploid","MLL","Other","Ph_like_CRLF2","Ph_like_non_CRLF2"]

stjude = pd.read_csv("ALL.StJude.sampXGenes.nsFilt.txt", sep="\t")
stjude.replace(['Hyperdiploid', 'Hypodiploid'], 'Aneuploid',inplace=True)
stjude.GroupCode.replace(5, 6,inplace=True)


# get sample metadata information
#sample_data = pd.read_csv("All.StJude.metadata.txt", sep= "\t")

# Randomly sample 75% the samples without replacement for training the models.
# Returns two Pandas DataFrames, 'test' and 'train'.


train, test = train_test_split(stjude, random_state=12)

#partion x and y variables for inputs to models
train_param = train.iloc[:,8:]
train_group = train.iloc[:,1]
test_param = test.iloc[:,8:]
test_group = test.iloc[:,1]

print("There are " + str(len(train_group)) + " samples in the training group and \nthere are " + str(len(test_group)) +" samples in the test group.\n")

print("We will use "+ str(len(train_param.columns)) + " expression values as input parameters.\n")

# set unit variance and mean to 0 to create a Gaussian standardized dataset.
# standardize using the training set distribution to avoid allowing the test 
# set to influence the model parameters.

z_scaler = StandardScaler().fit(train_param)
z_train_param = z_scaler.transform(train_param)
z_test_param = z_scaler.transform(test_param)

# Naive Bayes (Gaussian) Classifier
# fit model on training data, predict on test set, obtain accuracy and get 
# precision of model for each class.
naive = GaussianNB()
naive.fit(z_train_param, train_group)
pred_naive = naive.predict(z_test_param)
acc_naive = accuracy_score(test_group, pred_naive, normalize = True)
prec_naive = precision_score(test_group,pred_naive, average=None)

print("The accuracy of Naive Bayes Gaussian " + str(acc_naive) )
for i,j in zip(np.nditer(prec_naive), groups):
    print("The precision of Naive Bayes Gaussian for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")
##############################################################################
##############################################################################
##############################################################################

dtree = DecisionTreeClassifier()

dt_param = {
        "max_depth" : [None],
        "min_samples_split" : [2],
        "min_samples_leaf" : [1],
        "min_weight_fraction_leaf" : [0.],
        "max_features" : [None],
        "random_state" : [1],
        "max_leaf_nodes" : [None],
        "presort" : [True, False]
        }

dtreeCV = GridSearchCV(estimator = dtree, param_grid = dt_param,
                       scoring = None, n_jobs=-1,
                       cv=10, verbose = 1,
                       return_train_score = True
                       )
dtreeCV.fit(z_train_param, train_group)

dtree = dtreeCV.best_estimator_
pred_dtree = dtree.predict(z_test_param)
acc_dtree = accuracy_score(test_group, pred_dtree, normalize = True)
prec_dtree = precision_score(test_group,pred_dtree, average=None)
print("The accuracy of Decision Tree " + str(acc_dtree) )
for i,j in zip(np.nditer(prec_dtree), groups):
    print("The precision of Decision Tree for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
##############################################################################
##############################################################################

rf = RandomForestClassifier()

rf_param = {
        "n_estimators" : [1,10,50,100,250,500,750,1000],
        "criterion" : ["entropy", "gini"],
        "max_features" : ["auto"],
        "max_depth" : [None,1,5,10],
        "oob_score" : [False],
        "n_jobs" : [-1],
        "warm_start" : [False],
        "random_state" : [12]
        }

rfCV = GridSearchCV(estimator = rf, param_grid = rf_param,
                       scoring = None, n_jobs=-1,
                       cv=10, verbose = 1,
                       return_train_score = True
                       )

rfCV.fit(z_train_param,train_group)

rf = rfCV.best_estimator_
pred_rf = rf.predict(z_test_param)
acc_rf = accuracy_score(test_group, pred_rf, normalize = True)
prec_rf = precision_score(test_group,pred_rf, average=None)
print("The accuracy of Random Forest Classier " + str(acc_rf) )
for i,j in zip(np.nditer(prec_rf), groups):
    print("The precision of Random Forest Classifier for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
##############################################################################
##############################################################################


knn = KNeighborsClassifier()

knn_param = {
        "n_neighbors" : [i for i in range(1,50,5)],
        "weights" : ["uniform", "distance"],
        "algorithm" : ["ball_tree", "kd_tree"],
        "leaf_size" : [1,10,15,30,50],
        "p" : [1,2],
        "n_jobs" : [-1]
        }

knnCV = GridSearchCV(estimator = knn, param_grid = knn_param,
                     scoring = None, cv = 10, verbose = 1,
                     return_train_score = True
                     )

knnCV.fit(z_train_param, train_group)
knn = knnCV.best_estimator_
pred_knn = rf.predict(z_test_param)
acc_knn = accuracy_score(test_group, pred_knn, normalize = True)
prec_knn = precision_score(test_group,pred_knn, average=None)
print("The accuracy of KNN Classier : " + str(acc_knn) )
for i,j in zip(np.nditer(prec_knn), groups):
    print("The precision of KNN Classifier for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")


##############################################################################
##############################################################################
##############################################################################
##############################################################################

ridge = RidgeClassifierCV(alphas=[1e-3,1e-2,1e-1,0.5,1.0],cv=10).fit(z_train_param,train_group)
print("Ridge Classifier score train: " +ridge.score(z_train_param,train_group))

pred_ridge = ridge.predict(z_test_param)
print("Ridge Classifier score test: " + ridge.score(z_test_param, test_group))

acc_ridge = accuracy_score(test_group, pred_ridge, normalize = True)
prec_ridge = precision_score(test_group,pred_naive, average=None)

print("The accuracy of Ridge Classifier : " + str(acc_ridge) )
for i,j in zip(np.nditer(prec_ridge), groups):
    print("The precision of Ridge Classifier for ALL subtype " +j + ": "+ str(round(float(i),3) ))

print("\n\n\n")
print("##############################################################################")



##############################################################################
##############################################################################
##############################################################################
##############################################################################



logit_l1 = LogisticRegressionCV(cv=10, solver='saga', penalty = 'l1', 
                             n_jobs=-1,multi_class = 'multinomial',
                             random_state = 12).fit(z_train_param,train_group)


logit_l2 = LogisticRegressionCV(cv=10, solver='lbfgs', penalty = 'l2', 
                             n_jobs=-1,multi_class = 'multinomial',
                             random_state = 12).fit(z_train_param,train_group)

pred_logL1 = logit_l1.predict(z_test_param)

pred_logL2 = logit_l2.predict(z_test_param)

acc_logL1 = accuracy_score(test_group, pred_logL1, normalize = True)
prec_LogL1 = precision_score(test_group,pred_logL1, average=None)

acc_logL2 = accuracy_score(test_group, pred_logL2, normalize = True)
prec_LogL2 = precision_score(test_group,pred_logL2, average=None)

print("The accuracy of CV Logistic Regression (L1): " + str(acc_logL1) )
for i,j in zip(np.nditer(prec_LogL1), groups):
    print("The precision of CV Logistic Regression (L1) for ALL subtype " +j + ": "+ str(round(float(i),3) ))

print("The accuracy of CV Logistic Regression (L2): " + str(acc_logL2) )
for i,j in zip(np.nditer(prec_LogL2), groups):
    print("The precision of CV Logistic Regression (L2) for ALL subtype " +j + ": "+ str(round(float(i),3) ))

print("\n\n\n")
print("##############################################################################")


##############################################################################
##############################################################################
##############################################################################
##############################################################################

# precompute with alpha=1.0 (default, lasso) and keep foldids in order to 
# find optimal value of alpha.

warnings.filterwarnings('ignore')
elasticCV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, keep=True
                 )

warnings.filterwarnings('default')
cvglmnetPlot(elasticCV)
pred_elastic = cvglmnetPredict(elasticCV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_elastic = accuracy_score(test_group, pred_elastic, normalize = True)
prec_elastic = precision_score(test_group,pred_elastic, average=None)

print("The accuracy of CV Elastic Net Regression (a=1.0): " + str(acc_elastic) )
for i,j in zip(np.nditer(prec_LogL1), groups):
    print("The precision of CV Elastic Net Regression (a=1.0) for ALL subtype " +j + ": "+ str(round(float(i),3) ))

print("\n\n\n")
print("##############################################################################")


##############################################################################
##############################################################################
options = glmnetSet(alpha=0.95)
elastic95CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic95CV)
pred_95elastic = cvglmnetPredict(elastic95CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_95elastic = accuracy_score(test_group, pred_95elastic, normalize = True)
prec_95elastic = precision_score(test_group,pred_95elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.95): " + str(acc_95elastic) )
for i,j in zip(np.nditer(prec_95elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.95) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.9)
elastic09CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic09CV)
pred_09elastic = cvglmnetPredict(elastic09CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_09elastic = accuracy_score(test_group, pred_09elastic, normalize = True)
prec_09elastic = precision_score(test_group,pred_09elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.9): " + str(acc_09elastic) )
for i,j in zip(np.nditer(prec_09elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.9) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.8)
elastic08CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic08CV)
pred_08elastic = cvglmnetPredict(elastic08CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_08elastic = accuracy_score(test_group, pred_08elastic, normalize = True)
prec_08elastic = precision_score(test_group,pred_08elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.8): " + str(acc_08elastic) )
for i,j in zip(np.nditer(prec_08elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.8) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.7)
elastic07CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic07CV)
pred_07elastic = cvglmnetPredict(elastic07CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_07elastic = accuracy_score(test_group, pred_07elastic, normalize = True)
prec_07elastic = precision_score(test_group,pred_07elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.7): " + str(acc_07elastic) )
for i,j in zip(np.nditer(prec_07elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.7) for ALL subtype " +j + ": "+ str(round(float(i),3) ))

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.6)
elastic06CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic06CV)
pred_06elastic = cvglmnetPredict(elastic06CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_06elastic = accuracy_score(test_group, pred_06elastic, normalize = True)
prec_06elastic = precision_score(test_group,pred_06elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.6): " + str(acc_06elastic) )
for i,j in zip(np.nditer(prec_06elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.6) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.5)
elastic05CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic05CV)
pred_05elastic = cvglmnetPredict(elastic05CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_05elastic = accuracy_score(test_group, pred_05elastic, normalize = True)
prec_05elastic = precision_score(test_group,pred_05elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.5): " + str(acc_05elastic) )
for i,j in zip(np.nditer(prec_05elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.5) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################

options = glmnetSet(alpha=0.4)
elastic04CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic04CV)
pred_04elastic = cvglmnetPredict(elastic04CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_04elastic = accuracy_score(test_group, pred_04elastic, normalize = True)
prec_04elastic = precision_score(test_group,pred_04elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.4): " + str(acc_04elastic) )
for i,j in zip(np.nditer(prec_04elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.4) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.3)
elastic03CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic03CV)
pred_03elastic = cvglmnetPredict(elastic03CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_03elastic = accuracy_score(test_group, pred_03elastic, normalize = True)
prec_03elastic = precision_score(test_group,pred_03elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.3): " + str(acc_03elastic) )
for i,j in zip(np.nditer(prec_03elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.3) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.2)
elastic02CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic02CV)
pred_02elastic = cvglmnetPredict(elastic02CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_02elastic = accuracy_score(test_group, pred_02elastic, normalize = True)
prec_02elastic = precision_score(test_group,pred_02elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.2): " + str(acc_02elastic) )
for i,j in zip(np.nditer(prec_02elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.2) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
options = glmnetSet(alpha=0.1)
elastic01CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic01CV)
pred_01elastic = cvglmnetPredict(elastic01CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_01elastic = accuracy_score(test_group, pred_01elastic, normalize = True)
prec_01elastic= precision_score(test_group,pred_01elastic, average=None)
print("The accuracy of CV Elastic Net Regression (a=0.1): " + str(acc_01elastic) )
for i,j in zip(np.nditer(prec_01elastic), groups):
    print("The precision of CV Elastic Net Regression (a=0.1) for ALL subtype " +j + ": "+ str(round(float(i),3) ))
print("\n\n\n")
print("##############################################################################")

##############################################################################
##############################################################################
sys.__stdout__  = sys.stdout

log_file.close()