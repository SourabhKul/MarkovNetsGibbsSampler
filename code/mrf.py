import numpy as np
import json
import prob_table
import itertools
class mrf():
    '''This class represents an MRF model for the heart disease domain.
    you can use any internal data structures to build the model.
    The model must support the functions __init__(), conditional() 
    and estimate_marginal(). You can add any additional functions
    that you require. 
    '''
    
    def __init__(self, params):
        ''' function __init__(self,params)
        
            inputs: params -- a list containing the parameters for all CPTs in the model.
            Each element in the list represents a single CPT parameter as a dictionary 
            with three elements: "tv" is a dictionary with the target variable as the
            key and the value as the target value. "cvs" is a dictionary
            where the keys represent the parent (conditioning) variables, and the values
            represent the values of those variables. Finally, "p" is a float giving the
            probability that the parent variable takes its specified value given that
            the parent variables take their specified values. The first element of
            params is listed below. You may print params to further inspect its
            structure. Note: all strings in the params structure are unicode.
        
            example: [{'tv': {'A': '<45'}, 'p': 0.18518518518518517, 'cvs': {}},...
        
            outputs: None
        '''
        """ values each discrete node can take """
        global values
        values = {'A' : (1,2,3), 'G' : (1,2), 'CP' : (1,2,3,4), 'BP' : (1, 2),'CH' : (1,2), 'ECG' : (1,2),
        'HR' : (1,2), 'EIA' : (1,2), 'HD' : (1,2) }
        
        global alt_values
        alt_values = {'A' : ('>=55','45-55','<45'), 'G' : ('M','F'), 'CP' : ('Non-Anginal','Atypical','None','Typical'), 'BP' : ('H', 'L'),'CH' : ('H','L'), 'ECG' : ('Abnormal','Normal'),
        'HR' : ('H','L'), 'EIA' : ('Y','N'), 'HD' : ('Y','N') }

        """ order in which data is stored in the file """
        global order
        order = ['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR','EIA', 'HD']
        
        """ markov network defined as the markov blanket of each node """
        global mn
        mn = {}
        mn['A'] = ['G','HR', 'BP', 'CH','HD']
        mn['G'] = ['A', 'BP', 'CH']
        mn['CP'] = ['HD']
        mn['BP'] = ['A', 'G', 'CH', 'HD', 'HR']
        mn['CH'] = ['A', 'G', 'BP', 'HD']
        mn['ECG'] = ['HD']
        mn['HR'] = ['HD', 'BP', 'A']
        mn['EIA'] = ['HD']
        mn['HD'] = ['A', 'CH', 'ECG', 'HR', 'EIA', 'BP', 'CP']
        
        global factors, dim, phi, belongs_to, translations
        phi = {}
        belongs_to = {}
        dim = {}
        factors = {}
        factors[0] = ['A']
        factors[1] = ['A', 'CH', 'G']
        factors[2] = ['BP', 'G']
        factors[3] = ['CH', 'BP', 'HD']
        factors[4] = ['A','HR', 'BP', 'HD']
        factors[5] = ['CP', 'HD']
        factors[6] = ['EIA', 'HD']
        factors[7] = ['ECG', 'HD']
        factors[8] = ['G']
        
        belongs_to['A'] = [0,1,4]
        belongs_to['BP'] = [2,3,4]
        belongs_to['CH'] = [1,3]
        belongs_to['CP'] = [5]
        belongs_to['ECG'] = [7]
        belongs_to['EIA'] = [6]
        belongs_to['G'] = [1,2,8]
        belongs_to['HD'] = [3,4,5,6,7]
        belongs_to['HR'] = [4]

        translations = {'<45':0, 'Normal':0, 'L':0, 'F':0, 'N':0, 'Typical':0 , '45-55':1, 'Abnormal':1, 'H':1, 'M':1, 'Y':1, 'Atypical':1, '>=55':2, 'Non-Anginal':2,'None':3}                
        
        global one_set, two_set, three_set

        one_set = ['<45', 'Normal', 'L', 'F', 'N', 'Typical']
        two_set = ['45-55', 'Abnormal', 'H', 'M', 'Y', 'Atypical']
        three_set = ['>=55', 'Non-Angial']
        for i in factors.keys():
                dim[i] = []
                for mem in factors[i]:
                    dim[i].append(len(values[mem]))
                phi[i] = np.zeros(dim[i])
        
        for line in params:
            instance = dict(line['tv'].items() + line['cvs'].items())
            #print instance
            for node, value in instance.items():
                if value in one_set :
                    target_variable_value = 0
                
                elif value in two_set :
                    target_variable_value = 1

                elif value in three_set:
                    target_variable_value = 2

                else:
                    target_variable_value = 3
                instance[node] = target_variable_value
            value = line['p']
            #print instance, value
            if value == 0:
                print '*****************************value is zero'
            for i in factors.keys():
                #print instance.keys(), factors[i]
                if set(instance.keys()) == set(factors[i]):
                    argument = [0]*(len(factors[i]))
                    for key, valuez in instance.items():
                        argument[factors[i].index(key)]=valuez
                    #print argument , value
                    phi[i][tuple(argument)] = value
                    #print phi[i][tuple(argument)]
        
        #print phi
        pass

    def conditional(self, var, config):
        ''' function conditional(self,var,config)
        
            inputs: var - a string representing the name of a variable. 
                          For example, "A", or "HD".
                    config - a dictionary specifying the values of all of the variables
                             except for var. For example, if var="A", then
                             config might be {"ECG":"Normal","CH":"H","G":"F",...}
        
            outputs: a dictionary containing the conditional probability of each value
                     of var given the specified values of all of the other variables.
                     For example, if var="A", then the output might be
                     {'>=55': 0.3, '45-55': 0.2, '<45': 0.5}.
        '''        
        #print config
        for node, value in config.items():
            if value in one_set :
                target_variable_value = 0
            
            elif value in two_set :
                target_variable_value = 1

            elif value in three_set:
                target_variable_value = 2

            else:
                target_variable_value = 3
            config[node] = target_variable_value
        #print config
        cond_prob = {}
        cond_prob_num = {}
        cond_val_den = 0
        for val in (alt_values[var]):
            config[var] = translations[val]
            cond_prob_num[val] = 1
            for i in factors.keys():
                #print instance.keys(), factors[i]
                if i in belongs_to[var]:
                    argument = [0]*(len(factors[i]))
                    for key, valuez in config.items():
                        if key in factors[i]:
                            argument[factors[i].index(key)]=valuez
                    #print argument, var, factors[i], valuez, key
                    cond_prob_num[val] *= phi[i][tuple(argument)]
            cond_val_den += cond_prob_num[val]
            #print val, cond_val_den, cond_prob_num[val]       
        for val in (alt_values[var]):
            cond_prob[val] = cond_prob_num[val]/(cond_val_den)
        
        #print cond_prob

        return cond_prob

    def gibbs_query(self, targets, evidence, num_samples=1000):
        ''' gibbs_query(self, targets, evidence, num_samples=1000)
        
            inputs: evidence - a dictionary containing variable names and 
                            values to condition on. May contain any number
                            of variables, including being empty.
                            For example, {"HD":"N", "G":"M","HR":"L"}. 
        
                    targets - a list specifying the target variables for the query.
                            You may assume that variables are used as evidence or as
                            a targets, but not both.
        
                    num_samples - number of samples to use in the Gibbs sampler.
        
            output: an object of class prob_table that gives the approximate 
                    joint probability over the target variables after 
                    conditioning on the evidence. The approximation should be
                    computed using num_samples iterations of the Gibbs sampler.
                    You do not need to discard samples for burn-in. See the
                    prob_table class for more details.
        '''          
        #print evidence.keys()
        init_config = {}
        init_config.update(evidence)
        for key, vals in alt_values.items():
            if not (key in init_config.keys()):
                init_config.update({key : np.random.choice(vals)}) 
        
        #print init_config
        sample_list = []
        for n in range(num_samples):
            for key in init_config.keys():
                #print key, evidence.keys()
                if not key in evidence.keys():
                    init_config.pop(key)
                    #print key, init_config
                    sample_dist = self.conditional(key,init_config.copy())
                    sample = np.random.choice(sample_dist.keys(),p=sample_dist.values())
                    #print sample
                    init_config[key] = sample
            sample_list.append(init_config.copy())
        all_samples = prob_table.prob_table(sample_list)
        return all_samples