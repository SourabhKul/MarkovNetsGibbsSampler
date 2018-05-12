import numpy as np
from sklearn import linear_model

class mclrmiss():
    
    def __init__(self):
        ''' Add any initialization code required by your approach'''
    
    def fit(self, Xtr, Ytr):
        '''function fit(self, Xtr, Ytr)
            This functions fits the model to the training data.
            This requires estimating a multivariate normal distribution
            for P(X), and estimating a logistic regression model for 
            P(Y|X). You must implement the computation for the parameters
            of P(X) using numpy. For P(Y|X), you should use scikit-learn's
            implementation with the following parameters:
        
            LogisticRegression(C=1,multi_class="multinomial",solver="lbfgs")
        
            You will need to store the parameters of P(X) and the learned
            logistic regression model as member variables. This function
            has no outputs.
        
            inputs: Xtr - numpy array of feature vectors of shape (N,F) where
                     N is the number of data cases and F is the number of
                    features. No missing data.
        
                    Ytr - numpy array of labels of shape (N,)
        ''' 
        
        self.model_P_Y_X = linear_model.logistic.LogisticRegression(C=1,multi_class='multinomial',solver='lbfgs')
        self.model_P_Y_X.fit(Xtr,Ytr)
        self.params_P_Y_X = self.model_P_Y_X.get_params()
        self.mean_P_X = np.mean(Xtr,axis=0)
        self.cov_P_X = np.cov(Xtr,rowvar=False)
        #print ('mean', self.mean_P_X, 'covariance', self.cov_P_X)
        pass
    
    def predict_proba(self, Xte, reps):
        '''predict_proba(self, Xte, reps)
            This function approximates P(Y|x_on) for each data case x_on.
        
            inputs: Xte - numpy array of size (N,F) with nans indicating
                          missing entries. N is the number of data cases.
                          F is the number of features.
                    reps - the number of samples to use in the Monte Carlo 
                        approximation.
                    
            outputs: a numpy array of size (N,C) where each row 
                    corresponds to a data case from Xte and each
                    column is a class label. The value on row n 
                    and column c is the estimate of P(Y=c|x_on)
        '''
        #print(Xte)
        N,_ = np.shape(Xte)
        if N ==1:
            Xte.reshape(1,-1)
        predict_probas = np.zeros([N,13])
        for row in range(N):
            index_missing = np.argwhere(np.isnan(Xte[row]))
            index_obs = np.argwhere(~np.isnan(Xte[row]))
            
            if len(index_missing) != 0:
                mean_missing = self.mean_P_X[index_missing.T[0]]
                mean_obs = self.mean_P_X[index_obs.T[0]]
                
                sigma_m_m = self.cov_P_X[np.ix_(index_missing.T[0],index_missing.T[0])]
                sigma_m_o = self.cov_P_X[np.ix_(index_missing.T[0],index_obs.T[0])]
                sigma_o_m = self.cov_P_X[np.ix_(index_obs.T[0],index_missing.T[0])]
                sigma_o_o = self.cov_P_X[np.ix_(index_obs.T[0],index_obs.T[0])]

                #print(self.cov_P_X.shape)

                cond_mean = mean_missing + np.dot(np.dot(sigma_m_o,np.linalg.inv(sigma_o_o)),(Xte[row,index_obs.T[0]]-mean_obs))
                cond_covar = sigma_m_m - np.dot(np.dot(sigma_m_o,np.linalg.inv(sigma_o_o)),sigma_o_m)
                #print(cond_mean.shape, cond_covar.shape)
                predictions = np.zeros([reps,13])
                for sample in range(reps):
                    x_miss = np.random.multivariate_normal(cond_mean, cond_covar)
                    #print(row, sample, Xte[row])
                    Xte[row,index_missing.T[0]] = x_miss
                    #print(row, sample, Xte[row])
                    predictions[sample] = self.model_P_Y_X.predict_proba(Xte[row].reshape(1,-1))
                predict_probas[row] = (np.mean(predictions, axis=0))
            else:
                predict_probas[row] = self.model_P_Y_X.predict_proba(Xte[row].reshape(1,-1))
        return predict_probas
    
    def predict_proba_zis(self, Xte, reps):
        '''predict_proba(self, Xte, reps)
            This function approximates P(Y|x_on) for each data case x_on.
        
            inputs: Xte - numpy array of size (N,F) with nans indicating
                          missing entries. N is the number of data cases.
                          F is the number of features.
                    reps - the number of samples to use in the Monte Carlo 
                        approximation.
                    
            outputs: a numpy array of size (N,C) where each row 
                    corresponds to a data case from Xte and each
                    column is a class label. The value on row n 
                    and column c is the estimate of P(Y=c|x_on)
        '''
        
        N,_ = np.shape(Xte)
        predict_probas = np.zeros([N,13])
        for row in range(N):
            Xte[row] = np.nan_to_num(Xte[row])
            predict_probas[row] = self.model_P_Y_X.predict_proba(Xte[row].reshape(1,-1))
        return predict_probas
    

    def predict(self, Xte, reps):
        '''predict(self, Xte, reps)
            This function first approximates P(Y|x_on) for each data case x_on.
            It then returns the most likely lable according to the estimated 
            distribution P(Y|x_on).
        
            inputs: Xte - numpy array of size (N,F) with nans indicating
                          missing entries. N is the number of data cases.
                          F is the number of features.
                    reps - the number of samples to use in the Monte  Carlo 
                        approximation.
                    
            outputs: a numpy array of size (N,) giving the predictied class
                    values. The value in element n is the predictied class
                    for the n^th row of Xte based on P(Y|x_on).
        '''
        predict_probas = self.predict_proba(Xte,reps)
        predictions = np.argmax(predict_probas,axis=1)
        #print predictions
        return predictions

    def predict_zis(self, Xte, reps):
        '''predict(self, Xte, reps)
            This function first approximates P(Y|x_on) for each data case x_on.
            It then returns the most likely lable according to the estimated 
            distribution P(Y|x_on).
        
            inputs: Xte - numpy array of size (N,F) with nans indicating
                          missing entries. N is the number of data cases.
                          F is the number of features.
                    reps - the number of samples to use in the Monte  Carlo 
                        approximation.
                    
            outputs: a numpy array of size (N,) giving the predictied class
                    values. The value in element n is the predictied class
                    for the n^th row of Xte based on P(Y|x_on).
        '''
        predict_probas = self.predict_proba_zis(Xte,reps)
        predictions = np.argmax(predict_probas,axis=1)
        #print predictions
        return predictions

            
            