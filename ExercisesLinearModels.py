#Mathematics in Machine Learning - Exercises on Linear Models 
#Alessia Leclercq

import numpy as np
import pandas as pd
import seaborn as sns
import math
import random
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

#set state for reproducibility 
state = 5298
rng = np.random.RandomState(state)


#EXERCISE 1
df = pd.read_table("220330insulate.txt", names = ["insulation", "temperature", "consumption"], sep = '\s+')
df['insulation'] = df.insulation.apply(lambda x: 1 if x=='prima' else 0) 
df = df.astype({'insulation' : 'category'})

#Fitting a regression model for consumption with interaction lr_large
#Fitting a regression model fol cunsumption without interaction lr_small
lr_large = smf.ols(formula='consumption ~ C(insulation) + temperature + C(insulation)*temperature', data=df).fit()
lr_small = smf.ols(formula = 'consumption ~ C(insulation) + temperature', data =df).fit()

print(lr_large.summary())
print(lr_large.conf_int(alpha = 0.01, cols = None)) #90% CI for regression coefficients


#EXERCISE 2 
print(anova_lm(lr_small, lr_large))


#EXERCISE 3
new_df = pd.DataFrame({'insulation': [0, 1], 'temperature': [3.2, 3.2]})
new_df = new_df.astype({'insulation' : 'category'})

predictions = lr_large.get_prediction(new_df)
print(predictions.summary_frame(alpha = 0.01))


#EXERCISE 4 
x = rng.normal(0, 1, 100)
y = rng.normal(1+2*x, 0.1, 100)
df = pd.DataFrame({'x' : x, 'x_squared': x**2, 'y' : y})

lm_synthetic = smf.ols(formula = 'y ~ x + x_squared', data = df).fit()
print(lm_synthetic.summary())


#EXERCISE 5
df = pd.read_table("220420EXECSAL.txt" , sep = '\t')
df['LOG_SALARY'] = df['SALARY'].apply(lambda x: math.log(x)) 

#linear model for SALARY on all other features
lm_salary = smf.ols(formula = 'SALARY ~ EXP + EDUC + GENDER + NUMSUP + ASSETS + EXPSQ + GEN_SUP + LNSAL', data = df).fit()
#linear model for LOG_SALARY on all other features
lm_salary_log = smf.ols(formula = 'LOG_SALARY ~ EXP + EDUC + GENDER + NUMSUP + ASSETS + EXPSQ + GEN_SUP + LNSAL', data = df).fit()

residuals = lm_salary.resid
mean_residuals = residuals.mean() 

residuals_log = lm_salary_log.resid
mean_log_residuals = residuals_log.mean() 

#function to generate scatter plot for residual analysis
def scatter(x, y, mean, title, xlabel, ylabel): 
    ax = sns.scatterplot(x = x, y = y)
    ax.axhline(mean, color = 'c')
    ax.set(title = title, xlabel =xlabel, ylabel = ylabel)
    return 

scatter(df.SALARY, residuals, mean_residuals, 'Plot residuals - salary', 'SALARY', 'RESIDUALS')
scatter(df.LOG_SALARY, residuals_log, mean_log_residuals, 'Plot residuals - log_salary', 'SALARY', 'RESIDUALS')

#the B-coefficient for EXP is the odd increase in salary as the experience within the company increases of 1 year, while keeping all other features constant.
print(lm_salary_log.summary())


#EXERCISE 6
l = [[1, 202.4, 203.2, 223.7, 203.6],
[2, 242, 248.7, 259.8, 240.7],
[3, 220.4, 227.3, 240, 207.4],
[4, 230, 243.1, 247.7, 226.9],
[5, 191.6, 211.4, 218.7, 200.1],
[6, 247.7, 253, 268.1, 195.8],
[7, 214.8, 214.8, 233.9, 227.9],
[8, 245.4, 243.6, 257.8, 227.9],
[9, 224, 231.5, 238.2, 215.7],
[10, 252.2, 255.2, 265.4, 245.2]]

df_short = pd.DataFrame(l, columns = ['GOLFER', 'A', 'B', 'C', 'D'])
df = pd.melt(df_short, id_vars = 'GOLFER', value_vars = ['A', 'B', 'C', 'D'], var_name = 'BRAND', value_name = 'DISTANCE')

df = df.astype({'GOLFER': 'category'})

#Fit an additive model containing both BRAND and GOLFER as qualitative predictors for DISTANCE
lr_complete = smf.ols(formula='DISTANCE ~ C(BRAND) + C(GOLFER)', data=df).fit()
print(lr_complete.summary())

#Performing ANOVA test between the complete model and a model including only golfers to test for significance of BRAND as a whole
lr_golfers = smf.ols(formula='DISTANCE ~ C(GOLFER)', data=df).fit()
anova_lm(lr_golfers, lr_complete)

#Performing ANOVA test between the complete model and a model including only brands to test for significance of GOLFER as a whole
lr_brands = smf.ols(formula='DISTANCE ~ C(BRAND)', data=df).fit()
anova_lm(lr_brands, lr_complete)

#Calculate a 95% confidence interval for the mean distance difference between BRAND C and BRAND A. See lr_complete.summary() you get [9.703, 26.865]
#Calculate a 95% confidence interval for the mean distance difference between BRAND C and BRAND A without accounting for GOLFER
print(lr_brands.summary()) #[1.649, 34.911]



