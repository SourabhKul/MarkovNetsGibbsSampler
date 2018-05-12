import json
import mrf 

#Load the parameters
with open("../data/params.json", "r") as of:
        params = json.load(of)

#Create an mrf object
hd_mrf = mrf.mrf(params)
print hd_mrf.conditional("A" , {'ECG': u'Normal', 'CH': u'H', 'G': u'M', 'HR': u'L', 'EIA': u'N', 'BP': u'H', 'CP': u'None', 'HD': u'N'})
#Example of a simple query with no evidence
evidence    = {'CP':'Atypical'}
targets     = ["A"]
#cond = hd_mrf.conditional('A',{u'HR': u'L', u'BP': u'L', u'HD': u'Y', 'G' : 'M', 'ECG' : 'Abnormal', 'EIA' : 'Y', 'CP' : 'Atypical', 'CH' : 'H'})
q_2_7_a        = hd_mrf.gibbs_query(targets, evidence, 10000)
print "P(A='<45'|'CP':'Atypical'): ",q_2_7_a.get({"A":"<45"})
print "P(A='45-55'|'CP':'Atypical'): ",q_2_7_a.get({"A":"45-55"})
print "P(A='>=55'|'CP':'Atypical'): ",q_2_7_a.get({"A":">=55"})

evidence    = {'CH':'H', 'HR' : 'L'}
targets     = ["HD"]
q_2_7_b        = hd_mrf.gibbs_query(targets, evidence, 10000)
print "P(HD='Y'|'CH'='H', 'HR'='L'):",q_2_7_b.get({"HD":"Y"})
print "P(HD='N'|'CH'='H', 'HR'='L'):",q_2_7_b.get({"HD":"N"})

evidence    = {'HD':'N', 'HR' : 'L'}
targets     = ["ECG"]
q_2_7_c        = hd_mrf.gibbs_query(targets, evidence, 10000)
print "P(ECG='Abnormal'|'HD'='N', 'HR'='L'): ",q_2_7_c.get({"ECG":'Abnormal'})
print "P(ECG='Normal'|'HD'='N', 'HR'='L'): ",q_2_7_c.get({"ECG":'Normal'})

#alt_values = {'A' : ('>=55','45-55','<45'), 'G' : ('M','F'), 'CP' : ('Non-Anginal','Atypical','None','Typical'), 'BP' : ('H', 'L'),'CH' : ('H','L'), 'ECG' : ('Abnormal','Normal'),
#        'HR' : ('H','L'), 'EIA' : ('Y','N'), 'HD' : ('Y','N') }


evidence    = {'EIA':'N'}
targets     = ['CH', 'BP']
q_2_7_d        = hd_mrf.gibbs_query(targets, evidence, 10000)
print "P(CH='L',BP='L'|'EIA'='N'): ",q_2_7_d.get({"CH":'L', 'HR':'L'})
print "P(CH='L',BP='H'|'EIA'='N'): ",q_2_7_d.get({"CH":'L', 'HR':'H'})
print "P(CH='H',BP='L'|'EIA'='N'): ",q_2_7_d.get({"CH":'H', 'HR':'L'})
print "P(CH='H',BP='H'|'EIA'='N'): ",q_2_7_d.get({"CH":'H', 'HR':'H'})

evidence    = {'CP':'Typical', 'HR':'H'}
targets     = ['G', 'HD']
q_2_7_e        = hd_mrf.gibbs_query(targets, evidence, 10000)
print "P(G='M',HD='N'|'CP'='Typical', 'HR'='H'): ",q_2_7_e.get({"G":'M', 'HD':'N'})
print "P(G='M',HD='Y'|'CP'='Typical', 'HR'='H'): ",q_2_7_e.get({"G":'M', 'HD':'Y'})
print "P(G='F',HD='N'|'CP'='Typical', 'HR'='H'): ",q_2_7_e.get({"G":'F', 'HD':'N'})
print "P(G='F',HD='Y'|'CP'='Typical', 'HR'='H'): ",q_2_7_e.get({"G":'F', 'HD':'Y'})