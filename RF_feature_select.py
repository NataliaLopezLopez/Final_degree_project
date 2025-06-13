from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
import math

def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

file='matlab/all_filt.csv'
#file='matlab/select_filt.csv'
#file='matlab/probs_filt.csv'
data= pd.read_csv(file,header=None,sep='\,')
'''file='matlab/avSPE_filt.csv'
data2= pd.read_csv(file,header=None,sep='\t')
file='matlab/PSTPE_filt.csv'
data3= pd.read_csv(file,header=None,sep='\t')

data=pd.concat([data1,data2,data3],axis=1,ignore_index=True)'''

kf = KFold(n_splits=10)
rkf = RepeatedKFold(n_splits=10, n_repeats=20,random_state=2652125)


labels=[0]*107
labels.extend([1]*107)
labels=np.array(labels)

#X_new = SelectKBest(f_classif, k=8).fit_transform(data, labels)
#data=X_new

min_features_to_select = 1  # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
'''rfecv.fit(data, labels)
X_new=rfecv.fit_transform(data,labels)
#data=X_new
rfecv.get_support()

print(f"Optimal number of features: {rfecv.n_features_}")

n_scores = len(rfecv.cv_results_["mean_test_score"])
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    range(min_features_to_select, n_scores + min_features_to_select),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()'''

clf = RandomForestClassifier(n_estimators=55,max_depth=15, min_samples_split=50,max_features=.05, max_leaf_nodes=5,  min_samples_leaf=2)

sfs = SequentialFeatureSelector(clf, n_features_to_select=3,direction='forward',n_jobs=8)


sfs.fit(data, labels)
X_new=sfs.fit_transform(data,labels)

data=X_new

acc_train=[]; s_acc_train=[]
acc_test=[]; s_acc_test=[] 
f1_train=[]; s_f1_train=[]
f1_test=[]; s_f1_test=[]
rec_train=[]; s_rec_train=[]
rec_test=[]; s_rec_test=[] 
pre_train=[]; s_pre_train=[]
pre_test=[]; s_pre_test=[] 
spe_train=[]; s_spe_train=[]
spe_test=[]; s_spe_test=[] 

'''train_labels=[0]*train_size
train_labels.extend([1]*train_size)
test_labels=[0]*(107-train_size)
test_labels.extend([1]*(107-train_size))

train_select=random.sample(range(0, 107), train_size)


train_set=[data[0][i] for i in train_select]
train_set.extend([data[0][i+107] for i in train_select])

test_set=[data[0][i] for i in range(107) if i not in train_select]
test_set.extend([data[0][i+107] for i in range(107) if i not in train_select])
'''   

parameter=range(1)#range(5,155,5)

for p in parameter:

    TP_train=[]; FP_train=[]; FN_train=[]; TN_train=[]
    TP_test=[]; FP_test=[]; FN_test=[]; TN_test=[]

    for train, test in rkf.split(data[:107]):
        train_add=[i+107 for i in train]
        train = np.hstack([train,np.array(train_add)])

        test_add=[i+107 for i in test]
        test=np.hstack([test,np.array(test_add)])

        ''' test_set=np.transpose([data[0][test],data[1][test],data[2][test]])
        train_set=np.transpose([data[0][train],data[1][train],data[2][train]])
        '''
        
        test_set=data[test]
        train_set=data[train]

        train_labels=labels[train]
        test_labels=labels[test]

        #test_set=data.iloc[test]
        #train_set=data.iloc[train]


        clf = RandomForestClassifier(n_estimators=75,max_depth=15, min_samples_split=50,max_features=.05, max_leaf_nodes=5,  min_samples_leaf=2)

        clf = clf.fit(train_set, train_labels)
        predic_labels=clf.predict(test_set)
        predic_train_labels=clf.predict(train_set)


        TP_train.append(sum([1 for i in range(len(train_labels)) if  predic_train_labels[i]==1 and train_labels[i]==1 ]))
        FP_train.append(sum([1 for i in range(len(train_labels)) if  predic_train_labels[i]==1 and train_labels[i]==0 ]))
        FN_train.append(sum([1 for i in range(len(train_labels)) if  predic_train_labels[i]==0 and train_labels[i]==1 ]))
        TN_train.append(sum([1 for i in range(len(train_labels)) if  predic_train_labels[i]==0 and train_labels[i]==0 ]))



        TP_test.append(sum([1 for i in range(len(test_labels)) if  predic_labels[i]==1 and test_labels[i]==1 ]))
        FP_test.append(sum([1 for i in range(len(test_labels)) if  predic_labels[i]==1 and test_labels[i]==0 ]))
        FN_test.append(sum([1 for i in range(len(test_labels)) if  predic_labels[i]==0 and test_labels[i]==1 ]))
        TN_test.append(sum([1 for i in range(len(test_labels)) if  predic_labels[i]==0 and test_labels[i]==0 ]))



    acc_train.append(np.mean([(TP_train[i]+TN_train[i])/(TP_train[i]+TN_train[i]+FP_train[i]+FN_train[i]) for i in range(len(TP_train)) ]))
    s_acc_train.append(np.std([(TP_train[i]+TN_train[i])/(TP_train[i]+TN_train[i]+FP_train[i]+FN_train[i]) for i in range(len(TP_train)) ]))

    acc_test.append(np.mean([(TP_test[i]+TN_test[i])/(TP_test[i]+TN_test[i]+FP_test[i]+FN_test[i]) for i in range(len(TP_test)) ]))
    s_acc_test.append(np.std([(TP_test[i]+TN_test[i])/(TP_test[i]+TN_test[i]+FP_test[i]+FN_test[i]) for i in range(len(TP_test)) ]))
    
    f1_train.append(np.mean([2*TP_train[i]/(2*TP_train[i]+FP_train[i]+FN_train[i]) for i in  range(len(TP_train))]) )   
    s_f1_train.append(np.std([2*TP_train[i]/(2*TP_train[i]+FP_train[i]+FN_train[i]) for i in  range(len(TP_train))])  )  

    f1_test.append(np.mean([2*TP_test[i]/(2*TP_test[i]+FP_test[i]+FN_test[i]) for i in  range(len(TP_test))]) )   
    s_f1_test.append(np.std([2*TP_test[i]/(2*TP_test[i]+FP_test[i]+FN_test[i]) for i in  range(len(TP_test))])  )

    rec_train.append(np.mean([(TP_train[i])/(TP_train[i]+FN_train[i]) for i in range(len(TP_train)) ]))
    s_rec_train.append(np.std([(TP_train[i])/(TP_train[i]+FN_train[i])  for i in range(len(TP_train)) ]))

    rec_test.append(np.mean([(TP_test[i])/(TP_test[i]+FN_test[i]) for i in range(len(TP_test)) ]))
    s_rec_test.append(np.std([(TP_test[i])/(TP_test[i]+FN_test[i]) for i in range(len(TP_test)) ]))
    
    pre_train.append(np.mean([TP_train[i]/(TP_train[i]+FP_train[i]) for i in  range(len(TP_train))]) )   
    s_pre_train.append(np.std([TP_train[i]/(TP_train[i]+FP_train[i]) for i in  range(len(TP_train))])  )  

    pre_test.append(np.mean([TP_test[i]/(TP_test[i]+FP_test[i]) for i in  range(len(TP_test))]) )   
    s_pre_test.append(np.std([TP_test[i]/(TP_test[i]+FP_test[i]) for i in  range(len(TP_test))])  )

    spe_train.append(np.mean([TN_train[i]/(TN_train[i]+FP_train[i]) for i in  range(len(TP_train))]) )   
    s_spe_train.append(np.std([TN_train[i]/(TN_train[i]+FP_train[i]) for i in  range(len(TP_train))])  )  

    spe_test.append(np.mean([TN_test[i]/(TN_test[i]+FP_test[i]) for i in  range(len(TP_test))]) )   
    s_spe_test.append(np.std([TN_test[i]/(TN_test[i]+FP_test[i]) for i in  range(len(TP_test))])  )



'''erd=[acc_test[i]-s_acc_test[i] for i in range(len(acc_test))]
eru=[acc_test[i]+s_acc_test[i] for i in range(len(acc_test))]    
plt.plot(parameter, acc_test, 'r-')
plt.fill_between(parameter, erd, eru, color='r',alpha=.5)
erd=[acc_train[i]-s_acc_train[i] for i in range(len(acc_train))]
eru=[acc_train[i]+s_acc_train[i] for i in range(len(acc_train))]    
plt.plot(parameter, acc_train, 'b-')
plt.fill_between(parameter, erd, eru, color='b',alpha=.5)
plt.show()
print(max(acc_test))'''

print('Accuracy: '+str(round(acc_test[0],4))+' \pm '+str(round_up(s_acc_test[0],decimals=4)))
print('F1: '+str(round(f1_test[0],4))+' \pm '+str(round_up(s_f1_test[0],decimals=4)))
print('Precision: '+str(round(pre_test[0],4))+' \pm '+str(round_up(s_pre_test[0],decimals=4)))
print('Recall: '+str(round(rec_test[0],4))+ ' \pm '+str(round_up(s_rec_test[0],decimals=4)))
print('Specificity: '+str(round(spe_test[0],4))+' \pm '+str(round_up(s_spe_test[0],decimals=4)))

print(sfs.get_support())

'''plt.figure()
for i in range(12):
    plt.errorbar(i+1,np.mean(data[i][:107]),np.std(data[i][:107]),color='b',marker='o')
    plt.errorbar(i+.5,np.mean(data[i][107:]),np.std(data[i][107:]),color='r',marker='o')

plt.show()'''