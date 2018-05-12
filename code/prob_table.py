
class prob_table():
    '''This class represents a probability distribution
    over a set of variables as a table. It can use any
    internal member data structures. The only API method this 
    class must  include is get(). Only you will create objects 
    of this type. Autograding code will only ever call the get()
    method on prob_table objects returned by your code.
    '''    
    
    def __init__(self, sample_list, vars = None):
        '''function __init__(self,vars,...)
        
            inputs: vars - a list of names of variables that the
            distribution is defined over. For example, vars=["HD","BP"]
            
            outputs: None
            
            Notes: Include any additional arguments you need and 
            create any member variables needed to support your 
            implementation.
        '''
        self.sample_list = sample_list
        pass
    
    def get(self, config):
        '''function get(self, config)
        
            inputs: config - a Python dictionary where each key is 
            the name of a variable and each value is a value for that 
            variable. For example, config={"HD":"Y", "BP": "N:}.
        
            outputs: The probability stored in the table for the
            configuration config. For example, if vars=["HD","BP"],
            then the table represents P(HD,BP). If config={"HD":"Y", "BP": "N"},
            then the output should be P(HD=Y,BP=N). 
        '''
        count = 0.0
        num_samples = len(self.sample_list)
        for sample in self.sample_list:
            for key, value in config.items():
                #print key, sample[key], value
                if  sample[key] != value:
                    break
            else:
                count += 1 

        #print count, num_samples

        return float(count/num_samples)