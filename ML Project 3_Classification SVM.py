import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
from scipy.stats import uniform
from random import uniform as rndflt
from scipy.stats import randint
from matplotlib.artist import setp
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.feature_selection import mutual_info_classif,f_classif,chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_predict,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix,recall_score,f1_score,accuracy_score,precision_score
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.preprocessing import OrdinalEncoder



df = pd.read_csv('SAheart.csv')
df = df.drop(['row.names'], 1)

####################3##################identify categorical columns, if any############################
# for attribute in df.columns.tolist():
#     print('Unique values for {0} : {1}'.format(attribute, df[attribute].nunique()))

#####################################DEALING WITH TEXT CATEGORICAL DATA################################

'''since we only have 2 categories, it's essentialy binary and ordinal encoding should be sufficient'''
ordinal_encoder = OrdinalEncoder()
fam_hist = df[['famhist']]
famhist_encoded = ordinal_encoder.fit_transform(fam_hist)
famhist_encoded_df = pd.DataFrame(famhist_encoded)
famhist_encoded_df.rename(columns={0:'fam_hist'}, inplace=True)  #fam_hist is encoded famhist
#rejoining the encoded column to everything else
df = df.join(famhist_encoded_df) #honors index
df = df.drop(['famhist'], 1)
#print(df.head())

####################################### Sample a test set, put it aside, and never look at it##########
''' IMPORTANT!
We do not want to separate the X(inputs) from the y(label/target) before splitting into
test and train data if we are planning on having a stratfied split. This is becuase the 
attribute we're planning on stratify splitting on, will be missing from either X or y,
in this case y, becuase we want a balance of all wine qualities in both train and test 
data '''
train_set, test_set = train_test_split(df, test_size=0.25, shuffle=True, stratify=df['chd'])

########################################Scatter Matrix Visualizations################################################
# mpl.style.use('seaborn-dark')  
# #print(plt.style.available) # see what styles are available
# fig, ax = plt.subplots(1, 1)
# ax.set_facecolor('#F0F0F0')
# pd.plotting.scatter_matrix(train_set, alpha = 0.2, figsize = (20,20), ax=ax, diagonal = 'kde', c='#00473C')
# n = len(df.columns)
# axs = pd.plotting.scatter_matrix(train_set, alpha = 0.2, figsize = (8,8), ax=ax, diagonal = 'kde', c='#00473C')
# for x in range(n):
#     for y in range(n):
#         # to get the axis of subplots
#         ax = axs[x, y]
#         # to make x axis name vertical  
#         ax.xaxis.label.set_rotation(90)
#         # to make y axis name horizontal 
#         ax.yaxis.label.set_rotation(0)
#         # to make sure y axis names are outside the plot area
#         ax.yaxis.labelpad = 50
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
# fig = plt.gcf()
# fig.set_size_inches(15, 20, forward=True)
# fig.patch.set_facecolor('#F0F0F0')
# plt.show()
# ######################################Correlation Heatmap Visualization####################################
# sns.set(font_scale=0.8)
# correlation = train_set.corr()
# fig, ax = plt.subplots(1, 1)
# ax.set_facecolor('#F0F0F0')
# fig = plt.gcf()
# fig.set_size_inches(14, 10, forward=True)
# fig.patch.set_facecolor('#F0F0F0')
# # cmap = sns.palplot(sns.diverging_palette(220, 20, n=7))
# heatmap = sns.heatmap(correlation, annot=True, linewidths=1, linecolor='#F0F0F0', vmin=-0.9, vmax=1, cmap='BrBG')
# plt.show()

#######################################FEATURE SELECTION####################################################
'''We will use one of 2 methods, or both  For classifcation: Method 1, Correlation Method. Method 2. 
Univariate method using SelectKBest. '''
#Method1 was already carried out
#drop features based on f_classif below and collinearity above
train_set = train_set.drop(['obesity','alcohol','typea'],1)
test_set = test_set.drop(['obesity','alcohol','typea'],1)

#Good idea to split into X and y now as we'll only want study features vs target/feature here
X_train_set = train_set.drop(['chd'],1)
y_train_set = train_set['chd']
X_test_set = test_set.drop(['chd'],1)
y_test_set = test_set['chd']

#for scorng, we have the option between,chi2, f_classif, mutual_info_classif
#ANOVA will be used(explaned in detail on website, why)
'''mutual_info_classif'''
# bestfeatures = SelectKBest(score_func=f_classif, k=9)
# fit = bestfeatures.fit(X_train_set,y_train_set)
# df_scores = pd.DataFrame(fit.scores_)
# df_columns = pd.DataFrame(X_train_set.columns)
# # concatenate dataframes
# feature_scores = pd.concat([df_columns, df_scores],axis=1)
# feature_scores.columns = ['Feature_Name','Score']  # name output columns
# print(feature_scores.nlargest(9,'Score'))  # print 9 best features

########################################FEATURE SCALING##############################################
#dataset is pretty balanced so a standard scaler should be fine
scaler = StandardScaler()
X_train_set = scaler.fit_transform(X_train_set)
#separate scaling,equivalent to scaling only when you get the new data
X_test_set = scaler.fit_transform(X_test_set) 


#################################PERFORMANCE AND SCORING#############################################
'''WE CARE MORE ABOUT FN (so ROC curve), i.e it's more detrimental to classify someone as not having  
coronary heart disease whenthey do that it  is to classify someone as having chd when 
they dont'''
#we will aim to increase recall using class_weights but also keep precision decent

#############################SHORTLISITNG AND COMPARING METHODS#####################################

'''for this project, we are exploring SupportVectorMachines'''
'''we will compare a linear, polynomial and radal basis func. SVM classfier'''
'''this is equivalent to doing a gridsearch '''

# svc_linear = SVC(kernel='linear')
# svc_poly = SVC(kernel='poly')
# svc_rbf = SVC(kernel='rbf')

# for model in (svc_linear, svc_poly, svc_rbf):
#     print(model, cross_val_score(model, X_train_set, y_train_set, scoring="f1", cv=10).mean())
# #when you set cv=integer, integer, to specify the number of folds in a (Stratified)KFold,
# #if there arent enoung smaples in a class to be in each stratifed set, a Warning will be raised.


#######################################FINE TUNING HYPERPARAMETERS#######################################
svc_clf = SVC()
param_distributions = {'C':[1, 5, 10], 'kernel':['rbf', 'linear', 'poly'], 'degree':[3, 5, 7],
'tol':[5e-4, 1e-3, 5e-3], 'class_weight': [{0:0.5, 1:0.5}, {0:0.4, 1:0.6}, {0:0.6, 1:0.4}]}

'''GRIDSEARCH'''
def hypertuning_rscv(clf, p_distr, nbr_iter,X,y):
    rdmsearch = GridSearchCV(clf, param_grid=p_distr,
                                  n_jobs=-1, scoring='f1', cv=StratifiedShuffleSplit(n_splits=10))
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(svc_clf, param_distributions, 40, X_train_set, y_train_set)

print(rf_parameters)
print(rf_ht_score)

'''RESULTS'''
