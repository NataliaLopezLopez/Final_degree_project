from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import random
import math
import random


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

random.seed(10)

number_of_subjects=107

# FILES SELECTION
# Set files to train and test a RF classification. The script assumes two column files 
# of 'number_of_subjets' each, each file corresponding to a different experiment

file1 = 'eeg_processed/pspe_ver_L_3_lag_1_run_1_filt.csv'
file2 = 'eeg_processed/pspe_ver_L_3_lag_1_run_2_filt.csv'

################################################################################
data1 = pd.read_csv(file1,header=None,sep='\,')
data2 = pd.read_csv(file2,header=None,sep='\,')
data = pd.concat([data1,data2],axis=0,ignore_index=True)
################################################################################

# REPEATED K-FOLD SET-UP

kf = KFold(n_splits=10)
rkf = RepeatedKFold(n_splits=10, n_repeats=20,random_state=2652125)
#2652124

################################################################################


labels=[0]*number_of_subjects
labels.extend([1]*number_of_subjects)
labels=np.array(labels)


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


parameter=range(1)#range(2,100,2)
for p in parameter:

    TP_train=[]; FP_train=[]; FN_train=[]; TN_train=[]
    TP_test=[]; FP_test=[]; FN_test=[]; TN_test=[]

    for train, test in rkf.split(data[0][:number_of_subjects]):
        train_add=[i+number_of_subjects for i in train]
        train = np.hstack([train,np.array(train_add)])

        test_add=[i+number_of_subjects for i in test]
        test=np.hstack([test,np.array(test_add)])

        test_set=data.iloc[test]
        train_set=data.iloc[train]

        train_labels=labels[train]
        test_labels=labels[test]

        clf = RandomForestClassifier(n_estimators=75,max_depth=15, min_samples_split=50,max_features=None, max_leaf_nodes=10,  min_samples_leaf=2)
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


print('Accuracy: '+str(round(acc_test[0],4))+' \pm '+str(round_up(s_acc_test[0],decimals=4)))
print('F1: '+str(round(f1_test[0],4))+' \pm '+str(round_up(s_f1_test[0],decimals=4)))
print('Precision: '+str(round(pre_test[0],4))+' \pm '+str(round_up(s_pre_test[0],decimals=4)))
print('Recall: '+str(round(rec_test[0],4))+ ' \pm '+str(round_up(s_rec_test[0],decimals=4)))
print('Specificity: '+str(round(spe_test[0],4))+' \pm '+str(round_up(s_spe_test[0],decimals=4)))



