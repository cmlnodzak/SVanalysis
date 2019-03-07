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
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegressionCV
import scipy, importlib, pprint, warnings
from glmnetSet import glmnetSet
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPrint import cvglmnetPrint
from cvglmnetCoef import cvglmnetCoef
from cvglmentPredict import cvglmnetPredict


# get expression dataset for all genes and samples
stjude = pd.read_csv("Allgenes.phase1.StJude.txt", sep="\t")


# get sample metadata information
sample_data = pd.read_csv("All.StJude.metadata.txt", sep= "\t")

# Randomly sample 75% the samples without replacement for training the models.
# Returns two Pandas DataFrames, 'test' and 'train'.


train, test = train_test_split(stjude, random_state=12)

#partion x and y variables for inputs to models
train_param = train.iloc[:,3:]
train_group = train.iloc[:,2]
test_param = test.iloc[:,3:]
test_group = test.iloc[:,2]

print("There are " + len(train_group) + " samples in the training group and \nthere are " + len(test_group) +" samples in the test group.\n")

print("We will use "+ len(train_param.columns) + " expression values as input parameters.\n")

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
pre_naive_BCR = precision_score(test_group,pred_naive, pos_label = "BCR_ABL1")
pre_naive_E2A = precision_score(test_group,pred_naive, pos_label = "E2A_PBX1")
pre_naive_ERG = precision_score(test_group,pred_naive, pos_label = "ERG")
pre_naive_ETV = precision_score(test_group,pred_naive, pos_label = "ETV6_RUNX1")
pre_naive_hyper = precision_score(test_group,pred_naive, pos_label = "Hyperdiploid")
pre_naive_hypo = precision_score(test_group,pred_naive, pos_label = "Hypodiploid")
pre_naive_MLL = precision_score(test_group,pred_naive, pos_label = "MLL")
pre_naive_Other = precision_score(test_group,pred_naive, pos_label = "Other")
pre_naive_PhCRLF2 = precision_score(test_group,pred_naive, pos_label = "Ph_like_CRLF2")
pre_naive_PhNon = precision_score(test_group,pred_naive, pos_label = "Ph_like_non_CRLF2")



##############################################################################
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
pre_dtree_BCR = precision_score(test_group,pred_dtree, pos_label = "BCR_ABL1")
pre_dtree_E2A = precision_score(test_group,pred_dtree, pos_label = "E2A_PBX1")
pre_dtree_ERG = precision_score(test_group,pred_dtree, pos_label = "ERG")
pre_dtree_ETV = precision_score(test_group,pred_dtree, pos_label = "ETV6_RUNX1")
pre_dtree_hyper = precision_score(test_group,pred_dtree, pos_label = "Hyperdiploid")
pre_dtree_hypo = precision_score(test_group,pred_dtree, pos_label = "Hypodiploid")
pre_dtree_MLL = precision_score(test_group,pred_dtree, pos_label = "MLL")
pre_dtree_Other = precision_score(test_group,pred_dtree, pos_label = "Other")
pre_dtree_PhCRLF2 = precision_score(test_group,pred_dtree, pos_label = "Ph_like_CRLF2")
pre_dtree_PhNon = precision_score(test_group,pred_dtree, pos_label = "Ph_like_non_CRLF2")

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
pre_rf_BCR = precision_score(test_group,pred_rf, pos_label = "BCR_ABL1")
pre_rf_E2A = precision_score(test_group,pred_rf, pos_label = "E2A_PBX1")
pre_rf_ERG = precision_score(test_group,pred_rf, pos_label = "ERG")
pre_rf_ETV = precision_score(test_group,pred_rf, pos_label = "ETV6_RUNX1")
pre_rf_hyper = precision_score(test_group,pred_rf, pos_label = "Hyperdiploid")
pre_rf_hypo = precision_score(test_group,pred_rf, pos_label = "Hypodiploid")
pre_rf_MLL = precision_score(test_group,pred_rf, pos_label = "MLL")
pre_rf_Other = precision_score(test_group,pred_rf, pos_label = "Other")
pre_rf_PhCRLF2 = precision_score(test_group,pred_rf, pos_label = "Ph_like_CRLF2")
pre_rf_PhNon = precision_score(test_group,pred_rf, pos_label = "Ph_like_non_CRLF2")

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
pre_knn_BCR = precision_score(test_group,pred_knn, pos_label = "BCR_ABL1")
pre_knn_E2A = precision_score(test_group,pred_knn, pos_label = "E2A_PBX1")
pre_knn_ERG = precision_score(test_group,pred_knn, pos_label = "ERG")
pre_knn_ETV = precision_score(test_group,pred_knn, pos_label = "ETV6_RUNX1")
pre_knn_hyper = precision_score(test_group,pred_knn, pos_label = "Hyperdiploid")
pre_knn_hypo = precision_score(test_group,pred_knn, pos_label = "Hypodiploid")
pre_knn_MLL = precision_score(test_group,pred_knn, pos_label = "MLL")
pre_knn_Other = precision_score(test_group,pred_knn, pos_label = "Other")
pre_knn_PhCRLF2 = precision_score(test_group,pred_knn, pos_label = "Ph_like_CRLF2")
pre_knn_PhNon = precision_score(test_group,pred_knn, pos_label = "Ph_like_non_CRLF2")

##############################################################################
##############################################################################
##############################################################################
##############################################################################

ridge = RidgeClassifierCV(alphas=[1e-3,1e-2,1e-1,0.5,1.0],cv=10).fit(z_train_param,train_group)
print("Ridge Classifier score train: " +ridge.score(z_train_param,train_group))

pred_ridge = ridge.predict(z_test_param)
print("Ridge Classifier score test: " + ridge.score(z_test_param, test_group))

acc_ridge = accuracy_score(test_group, pred_ridge, normalize = True)
pre_ridge_BCR = precision_score(test_group,pred_ridge, pos_label = "BCR_ABL1")
pre_ridge_E2A = precision_score(test_group,pred_ridge, pos_label = "E2A_PBX1")
pre_ridge_ERG = precision_score(test_group,pred_ridge, pos_label = "ERG")
pre_ridge_ETV = precision_score(test_group,pred_ridge, pos_label = "ETV6_RUNX1")
pre_ridge_hyper = precision_score(test_group,pred_ridge, pos_label = "Hyperdiploid")
pre_ridge_hypo = precision_score(test_group,pred_ridge, pos_label = "Hypodiploid")
pre_ridge_MLL = precision_score(test_group,pred_ridge, pos_label = "MLL")
pre_ridge_Other = precision_score(test_group,pred_ridge, pos_label = "Other")
pre_ridge_PhCRLF2 = precision_score(test_group,pred_ridge, pos_label = "Ph_like_CRLF2")
pre_ridge_PhNon = precision_score(test_group,pred_ridge, pos_label = "Ph_like_non_CRLF2")

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
pre_logL1_BCR = precision_score(test_group,pred_logL1, pos_label = "BCR_ABL1")
pre_logL1_E2A = precision_score(test_group,pred_logL1, pos_label = "E2A_PBX1")
pre_logL1_ERG = precision_score(test_group,pred_logL1, pos_label = "ERG")
pre_logL1_ETV = precision_score(test_group,pred_logL1, pos_label = "ETV6_RUNX1")
pre_logL1_hyper = precision_score(test_group,pred_logL1, pos_label = "Hyperdiploid")
pre_logL1_hypo = precision_score(test_group,pred_logL1, pos_label = "Hypodiploid")
pre_logL1_MLL = precision_score(test_group,pred_logL1, pos_label = "MLL")
pre_logL1_Other = precision_score(test_group,pred_logL1, pos_label = "Other")
pre_logL1_PhCRLF2 = precision_score(test_group,pred_logL1, pos_label = "Ph_like_CRLF2")
pre_logL1_PhNon = precision_score(test_group,pred_logL1, pos_label = "Ph_like_non_CRLF2")

acc_logL2 = accuracy_score(test_group, pred_logL2, normalize = True)
pre_logL2_BCR = precision_score(test_group,pred_logL2, pos_label = "BCR_ABL1")
pre_logL2_E2A = precision_score(test_group,pred_logL2, pos_label = "E2A_PBX1")
pre_logL2_ERG = precision_score(test_group,pred_logL2, pos_label = "ERG")
pre_logL2_ETV = precision_score(test_group,pred_logL2, pos_label = "ETV6_RUNX1")
pre_logL2_hyper = precision_score(test_group,pred_logL2, pos_label = "Hyperdiploid")
pre_logL2_hypo = precision_score(test_group,pred_logL2, pos_label = "Hypodiploid")
pre_logL2_MLL = precision_score(test_group,pred_logL2, pos_label = "MLL")
pre_logL2_Other = precision_score(test_group,pred_logL2, pos_label = "Other")
pre_logL2_PhCRLF2 = precision_score(test_group,pred_logL2, pos_label = "Ph_like_CRLF2")
pre_logL2_PhNon = precision_score(test_group,pred_logL2, pos_label = "Ph_like_non_CRLF2")

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
pre_elastic_BCR = precision_score(test_group,pred_elastic, pos_label = "BCR_ABL1")
pre_elastic_E2A = precision_score(test_group,pred_elastic, pos_label = "E2A_PBX1")
pre_elastic_ERG = precision_score(test_group,pred_elastic, pos_label = "ERG")
pre_elastic_ETV = precision_score(test_group,pred_elastic, pos_label = "ETV6_RUNX1")
pre_elastic_hyper = precision_score(test_group,pred_elastic, pos_label = "Hyperdiploid")
pre_elastic_hypo = precision_score(test_group,pred_elastic, pos_label = "Hypodiploid")
pre_elastic_MLL = precision_score(test_group,pred_elastic, pos_label = "MLL")
pre_elastic_Other = precision_score(test_group,pred_elastic, pos_label = "Other")
pre_elastic_PhCRLF2 = precision_score(test_group,pred_elastic, pos_label = "Ph_like_CRLF2")
pre_elastic_PhNon = precision_score(test_group,pred_elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.95)
elastic95CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic95CV)
pred_95elastic = cvglmnetPredict(elastic95CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_95elastic = accuracy_score(test_group, pred_95elastic, normalize = True)
pre_95elastic_BCR = precision_score(test_group,pred_95elastic, pos_label = "BCR_ABL1")
pre_95elastic_E2A = precision_score(test_group,pred_95elastic, pos_label = "E2A_PBX1")
pre_95elastic_ERG = precision_score(test_group,pred_95elastic, pos_label = "ERG")
pre_95elastic_ETV = precision_score(test_group,pred_95elastic, pos_label = "ETV6_RUNX1")
pre_95elastic_hyper = precision_score(test_group,pred_95elastic, pos_label = "Hyperdiploid")
pre_95elastic_hypo = precision_score(test_group,pred_95elastic, pos_label = "Hypodiploid")
pre_95elastic_MLL = precision_score(test_group,pred_95elastic, pos_label = "MLL")
pre_95elastic_Other = precision_score(test_group,pred_95elastic, pos_label = "Other")
pre_95elastic_PhCRLF2 = precision_score(test_group,pred_95elastic, pos_label = "Ph_like_CRLF2")
pre_95elastic_PhNon = precision_score(test_group,pred_95elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.9)
elastic09CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic09CV)
pred_09elastic = cvglmnetPredict(elastic09CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_09elastic = accuracy_score(test_group, pred_09elastic, normalize = True)
pre_09elastic_BCR = precision_score(test_group,pred_09elastic, pos_label = "BCR_ABL1")
pre_09elastic_E2A = precision_score(test_group,pred_09elastic, pos_label = "E2A_PBX1")
pre_09elastic_ERG = precision_score(test_group,pred_09elastic, pos_label = "ERG")
pre_09elastic_ETV = precision_score(test_group,pred_09elastic, pos_label = "ETV6_RUNX1")
pre_09elastic_hyper = precision_score(test_group,pred_09elastic, pos_label = "Hyperdiploid")
pre_09elastic_hypo = precision_score(test_group,pred_09elastic, pos_label = "Hypodiploid")
pre_09elastic_MLL = precision_score(test_group,pred_09elastic, pos_label = "MLL")
pre_09elastic_Other = precision_score(test_group,pred_09elastic, pos_label = "Other")
pre_09elastic_PhCRLF2 = precision_score(test_group,pred_09elastic, pos_label = "Ph_like_CRLF2")
pre_09elastic_PhNon = precision_score(test_group,pred_09elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.8)
elastic08CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic08CV)
pred_08elastic = cvglmnetPredict(elastic08CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_08elastic = accuracy_score(test_group, pred_08elastic, normalize = True)
pre_08elastic_BCR = precision_score(test_group,pred_08elastic, pos_label = "BCR_ABL1")
pre_08elastic_E2A = precision_score(test_group,pred_08elastic, pos_label = "E2A_PBX1")
pre_08elastic_ERG = precision_score(test_group,pred_08elastic, pos_label = "ERG")
pre_08elastic_ETV = precision_score(test_group,pred_08elastic, pos_label = "ETV6_RUNX1")
pre_08elastic_hyper = precision_score(test_group,pred_08elastic, pos_label = "Hyperdiploid")
pre_08elastic_hypo = precision_score(test_group,pred_08elastic, pos_label = "Hypodiploid")
pre_08elastic_MLL = precision_score(test_group,pred_08elastic, pos_label = "MLL")
pre_08elastic_Other = precision_score(test_group,pred_08elastic, pos_label = "Other")
pre_08elastic_PhCRLF2 = precision_score(test_group,pred_08elastic, pos_label = "Ph_like_CRLF2")
pre_08elastic_PhNon = precision_score(test_group,pred_08elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.7)
elastic07CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic07CV)
pred_07elastic = cvglmnetPredict(elastic07CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_07elastic = accuracy_score(test_group, pred_07elastic, normalize = True)
pre_07elastic_BCR = precision_score(test_group,pred_07elastic, pos_label = "BCR_ABL1")
pre_07elastic_E2A = precision_score(test_group,pred_07elastic, pos_label = "E2A_PBX1")
pre_07elastic_ERG = precision_score(test_group,pred_07elastic, pos_label = "ERG")
pre_07elastic_ETV = precision_score(test_group,pred_07elastic, pos_label = "ETV6_RUNX1")
pre_07elastic_hyper = precision_score(test_group,pred_07elastic, pos_label = "Hyperdiploid")
pre_07elastic_hypo = precision_score(test_group,pred_07elastic, pos_label = "Hypodiploid")
pre_07elastic_MLL = precision_score(test_group,pred_07elastic, pos_label = "MLL")
pre_07elastic_Other = precision_score(test_group,pred_07elastic, pos_label = "Other")
pre_07elastic_PhCRLF2 = precision_score(test_group,pred_07elastic, pos_label = "Ph_like_CRLF2")
pre_07elastic_PhNon = precision_score(test_group,pred_07elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.6)
elastic06CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic06CV)
pred_06elastic = cvglmnetPredict(elastic06CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_06elastic = accuracy_score(test_group, pred_06elastic, normalize = True)
pre_06elastic_BCR = precision_score(test_group,pred_06elastic, pos_label = "BCR_ABL1")
pre_06elastic_E2A = precision_score(test_group,pred_06elastic, pos_label = "E2A_PBX1")
pre_06elastic_ERG = precision_score(test_group,pred_06elastic, pos_label = "ERG")
pre_06elastic_ETV = precision_score(test_group,pred_06elastic, pos_label = "ETV6_RUNX1")
pre_06elastic_hyper = precision_score(test_group,pred_06elastic, pos_label = "Hyperdiploid")
pre_06elastic_hypo = precision_score(test_group,pred_06elastic, pos_label = "Hypodiploid")
pre_06elastic_MLL = precision_score(test_group,pred_06elastic, pos_label = "MLL")
pre_06elastic_Other = precision_score(test_group,pred_06elastic, pos_label = "Other")
pre_06elastic_PhCRLF2 = precision_score(test_group,pred_06elastic, pos_label = "Ph_like_CRLF2")
pre_06elastic_PhNon = precision_score(test_group,pred_06elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.5)
elastic05CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic05CV)
pred_05elastic = cvglmnetPredict(elastic05CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_05elastic = accuracy_score(test_group, pred_05elastic, normalize = True)
pre_05elastic_BCR = precision_score(test_group,pred_05elastic, pos_label = "BCR_ABL1")
pre_05elastic_E2A = precision_score(test_group,pred_05elastic, pos_label = "E2A_PBX1")
pre_05elastic_ERG = precision_score(test_group,pred_05elastic, pos_label = "ERG")
pre_05elastic_ETV = precision_score(test_group,pred_05elastic, pos_label = "ETV6_RUNX1")
pre_05elastic_hyper = precision_score(test_group,pred_05elastic, pos_label = "Hyperdiploid")
pre_05elastic_hypo = precision_score(test_group,pred_05elastic, pos_label = "Hypodiploid")
pre_05elastic_MLL = precision_score(test_group,pred_05elastic, pos_label = "MLL")
pre_05elastic_Other = precision_score(test_group,pred_05elastic, pos_label = "Other")
pre_05elastic_PhCRLF2 = precision_score(test_group,pred_05elastic, pos_label = "Ph_like_CRLF2")
pre_05elastic_PhNon = precision_score(test_group,pred_05elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.4)
elastic04CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic04CV)
pred_04elastic = cvglmnetPredict(elastic04CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_04elastic = accuracy_score(test_group, pred_04elastic, normalize = True)
pre_04elastic_BCR = precision_score(test_group,pred_04elastic, pos_label = "BCR_ABL1")
pre_04elastic_E2A = precision_score(test_group,pred_04elastic, pos_label = "E2A_PBX1")
pre_04elastic_ERG = precision_score(test_group,pred_04elastic, pos_label = "ERG")
pre_04elastic_ETV = precision_score(test_group,pred_04elastic, pos_label = "ETV6_RUNX1")
pre_04elastic_hyper = precision_score(test_group,pred_04elastic, pos_label = "Hyperdiploid")
pre_04elastic_hypo = precision_score(test_group,pred_04elastic, pos_label = "Hypodiploid")
pre_04elastic_MLL = precision_score(test_group,pred_04elastic, pos_label = "MLL")
pre_04elastic_Other = precision_score(test_group,pred_04elastic, pos_label = "Other")
pre_04elastic_PhCRLF2 = precision_score(test_group,pred_04elastic, pos_label = "Ph_like_CRLF2")
pre_04elastic_PhNon = precision_score(test_group,pred_04elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.3)
elastic03CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic03CV)
pred_03elastic = cvglmnetPredict(elastic03CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_03elastic = accuracy_score(test_group, pred_03elastic, normalize = True)
pre_03elastic_BCR = precision_score(test_group,pred_03elastic, pos_label = "BCR_ABL1")
pre_03elastic_E2A = precision_score(test_group,pred_03elastic, pos_label = "E2A_PBX1")
pre_03elastic_ERG = precision_score(test_group,pred_03elastic, pos_label = "ERG")
pre_03elastic_ETV = precision_score(test_group,pred_03elastic, pos_label = "ETV6_RUNX1")
pre_03elastic_hyper = precision_score(test_group,pred_03elastic, pos_label = "Hyperdiploid")
pre_03elastic_hypo = precision_score(test_group,pred_03elastic, pos_label = "Hypodiploid")
pre_03elastic_MLL = precision_score(test_group,pred_03elastic, pos_label = "MLL")
pre_03elastic_Other = precision_score(test_group,pred_03elastic, pos_label = "Other")
pre_03elastic_PhCRLF2 = precision_score(test_group,pred_03elastic, pos_label = "Ph_like_CRLF2")
pre_03elastic_PhNon = precision_score(test_group,pred_03elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.2)
elastic02CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic02CV)
pred_02elastic = cvglmnetPredict(elastic02CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_02elastic = accuracy_score(test_group, pred_02elastic, normalize = True)
pre_02elastic_BCR = precision_score(test_group,pred_02elastic, pos_label = "BCR_ABL1")
pre_02elastic_E2A = precision_score(test_group,pred_02elastic, pos_label = "E2A_PBX1")
pre_02elastic_ERG = precision_score(test_group,pred_02elastic, pos_label = "ERG")
pre_02elastic_ETV = precision_score(test_group,pred_02elastic, pos_label = "ETV6_RUNX1")
pre_02elastic_hyper = precision_score(test_group,pred_02elastic, pos_label = "Hyperdiploid")
pre_02elastic_hypo = precision_score(test_group,pred_02elastic, pos_label = "Hypodiploid")
pre_02elastic_MLL = precision_score(test_group,pred_02elastic, pos_label = "MLL")
pre_02elastic_Other = precision_score(test_group,pred_02elastic, pos_label = "Other")
pre_02elastic_PhCRLF2 = precision_score(test_group,pred_02elastic, pos_label = "Ph_like_CRLF2")
pre_02elastic_PhNon = precision_score(test_group,pred_02elastic, pos_label = "Ph_like_non_CRLF2")

options = glmnetSet(alpha=0.1)
elastic01CV = cvglmnet(x = z_train_param.copy(), y = train_group.copy(),
                 family="multinomial", mtype ="grouped", parallel = -1, foldid=elasticCV.foldid
                 )
cvglmnetPlot(elastic01CV)
pred_01elastic = cvglmnetPredict(elastic01CV, newx = z_test_param, s= 'lambda_min', ptype = "class")
acc_01elastic = accuracy_score(test_group, pred_01elastic, normalize = True)
pre_01elastic_BCR = precision_score(test_group,pred_01elastic, pos_label = "BCR_ABL1")
pre_01elastic_E2A = precision_score(test_group,pred_01elastic, pos_label = "E2A_PBX1")
pre_01elastic_ERG = precision_score(test_group,pred_01elastic, pos_label = "ERG")
pre_01elastic_ETV = precision_score(test_group,pred_01elastic, pos_label = "ETV6_RUNX1")
pre_01elastic_hyper = precision_score(test_group,pred_01elastic, pos_label = "Hyperdiploid")
pre_01elastic_hypo = precision_score(test_group,pred_01elastic, pos_label = "Hypodiploid")
pre_01elastic_MLL = precision_score(test_group,pred_01elastic, pos_label = "MLL")
pre_01elastic_Other = precision_score(test_group,pred_01elastic, pos_label = "Other")
pre_01elastic_PhCRLF2 = precision_score(test_group,pred_01elastic, pos_label = "Ph_like_CRLF2")
pre_01elastic_PhNon = precision_score(test_group,pred_01elastic, pos_label = "Ph_like_non_CRLF2")


