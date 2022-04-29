#Alessia Leclercq Logistic Regression 
#Tutorial for UCLA in Python

import pandas as pd
import numpy as np
from math import exp
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial


df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
print(df.head())
print(df.describe())

#count frequencies for different rank values according to admit
print(df.filter(['admit','rank', 'gpa']).groupby(['admit', 'rank']).count())


#transforming rank into categorical ordered variable
df = df.astype({'rank':'category'})

#Get dummies
df = pd.get_dummies(df)
df.drop('rank_1', axis =1 , inplace = True)
print(df.head())

endog = df.admit #Y vector
exog = df.drop('admit', axis = 1) #X matrix
exog = sm.add_constant(exog, prepend=False) #adding the constant column (not automatically performed)

model = GLM(endog, exog, family = Binomial()).fit()
print(model.summary2()) #confidence intervals like the R equivalent confint.default(model), hence using standard errors

#test for the overall effect of rank on the model 
hypothesis_0 = '(rank_2 = rank_3 = rank_4 = 0)'
print(model.wald_test(hypothesis_0))

#test for the difference between two coefficients (check for coefficient:rank_2 = coefficient:rank_3)
hypothesis_1 = '(rank_2 = rank_3)'
print(model.wald_test(hypothesis_1))

#exponentiation of coefficients and confidence intervals 
def exponentiate(x):
    return x.apply(lambda element: exp(element))

params = model.params
ci = model.conf_int(alpha = 0.05)
                    
exp_results= pd.concat([params, ci], axis = 1)
exp_results.columns = ['coefficients', '0.025', '0.975']
exp_results = exp_results.apply(lambda x: exponentiate(x))

print(exp_results)


#prediction of probabilities
mean_gre = np.ones(4)*df.gre.mean()
mean_gpa = np.ones(4)*df.gpa.mean()
rank = [1,2,3,4]

new_df = pd.DataFrame({'gre': mean_gre, 'gpa' : mean_gpa, 'rank': rank}).astype({'rank':'category'})
new_df = pd.get_dummies(new_df)
new_df.drop('rank_1', axis =1 , inplace = True)

new_df['const'] = np.ones(4)
new_df['rankP'] = model.predict(exog = new_df)
print(new_df)


