import numpy as np
import mclrmiss

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

Xtr = np.load("../data/xtrain.npy")
Xte = np.load("../data/xtest.npy")
Ytr = np.load("../data/ytrain.npy")
Yte = np.load("../data/ytest.npy")

L_Yte,F = Xte.shape
print '***   Data Loaded, Fitting...   ***'
#Note: to speed testing, you can run on
#a subset of data and use fewer samples. 
#For final experiments, use all data cases.
clf = mclrmiss.mclrmiss()
clf.fit(Xtr[:,:],Ytr[:])
#probs = clf.predict_proba(Xte,10)
print '***  Fitting Complete, Predicting...   ***'
count = 0.0
log_likelihood = 0.0
predictions = clf.predict(Xte,10)
probas = clf.predict_proba(Xte,10)
print '***  Prediction Complete, Scoring...   ***'
for sample in range(L_Yte):
    #print Yte[sample], predictions[sample] 
    if Yte[sample] == predictions[sample]:
        count += 1
    log_likelihood += np.log(probas[sample,Yte[sample]])
print 'Accuracy_sampling: ',(float(count)/float(len(Yte)))*100,'%', count, len(Yte), 'avg_con_log_likelihood', float(log_likelihood/len(Yte))
count = 0.0
log_likelihood = 0.0
predictions = clf.predict_zis(Xte,10)
probas = clf.predict_proba_zis(Xte,10)
print '***  Prediction Complete, Scoring...   ***'
for sample in range(L_Yte):
    #print Yte[sample], predictions[sample] 
    if Yte[sample] == predictions[sample]:
        count += 1
    log_likelihood += np.log(probas[sample,Yte[sample]])
print 'Accuracy_zero_imputation: ',(float(count)/float(len(Yte)))*100,'%', count, len(Yte), 'avg_con_log_likelihood', float(log_likelihood/len(Yte))

Xtr = np.load("../data/xtrain.npy")
Xte = np.load("../data/xtest.npy")
Ytr = np.load("../data/ytrain.npy")
Yte = np.load("../data/ytest.npy")


print F
Xte_len = [None]*(F)
Yte_len = [None]*(F)
for l in range(F):
    #print l
    i = 0
    Xte_len[l] = [None]*(L_Yte)
    Yte_len[l] = [None]*(L_Yte)
    for row in range(L_Yte):
        #print(Xte[row], (Xte[row]== np.nan).sum())
        if l == sum(np.isnan(Xte[row])):
            Xte_len[l][i] = Xte[row]
            Yte_len[l][i] = Yte[row]
            i += 1
    #print(Xte_len[l][:i])
    if i != 0:
        count = 0.0
        log_likelihood = 0.0
        predictions = clf.predict(np.array(Xte_len[l][:i]),10)
        probas = clf.predict_proba(np.array(Xte_len[l][:i]),10)
        #print '***  Prediction Complete, Scoring...   ***'
        for sample in range(len(Xte_len[l][:i])):
            #print Yte[sample], predictions[sample] 
            if Yte_len[l][sample] == predictions[sample]:
                count += 1
            log_likelihood += np.log(probas[sample,predictions[sample]])
        print 'Sampling,',l,',',(float(count)/float(len(Yte_len[l][:i])))*100, ',', float(log_likelihood/len(Yte_len[l][:i]))
        count = 0.0
        log_likelihood = 0.0
        predictions = clf.predict_zis(np.array(Xte_len[l][:i]),10)
        probas = clf.predict_proba_zis(np.array(Xte_len[l][:i]),10)
        #print '***  Prediction Complete, Scoring...   ***'
        for sample in range(len(Xte_len[l][:i])):
            #print Yte[sample], predictions[sample] 
            if Yte_len[l][sample] == predictions[sample]:
                count += 1
            log_likelihood += np.log(probas[sample,predictions[sample]])
        print 'Zero_imputatation,',l,',',(float(count)/float(len(Yte_len[l][:i])))*100, ',', float(log_likelihood/len(Yte_len[l][:i]))