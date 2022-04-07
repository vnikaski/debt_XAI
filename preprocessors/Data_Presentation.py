#!/usr/bin/env python
# coding: utf-8

# In[550]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


import warnings
warnings.filterwarnings("ignore")


# In[551]:


df = pd.read_csv ('/Users/ja/Downloads/temat_3_dane.csv')
x = df.loc[:, df.columns != 'Y']
Y = df.loc[:, df.columns == 'Y']
df_expenses = df.iloc[:, :246]
df_income = df.iloc[:,247:]


# ## Initial analysis

# In[552]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    
    if tt[tt['Total']!=0].empty:
        print('There is no missing values in the whole dataset.')
    else:    
        tt = tt[tt['Total']!=0]
        print(f'There is {len(tt.index)} missing rows')
        return(tt)

def zero_values(data):
    count=0
    total=0
    for column_name in x.columns:
        column = x[column_name]
        count += (column == 0).sum()
    return count/((len(x.columns)-1)*len(x.index))*100

#Output
print(f"There is {len(x.columns)} variables.")
missing_data(x)
print(f'There is {format(zero_values(x),".2f")}% of zero values. ')
balance=(len(df[df['Y']==1].index)/len(df.index))*100
print(f'Only {format(balance,".2f")}% of people in the dataset did not pay off the loan. Which means that the data is higly inbalanced.')

sns.countplot(df['Y'], palette='Set3')
sns.despine()


# As there is 294 variables, it is impossible to present and analyse all of them with one take. Therefore it is nessesary to understand concept of those variables and group them in order to have better picture about the problem.
# 
# All variables can be devided into 42 categories:
# 
# Categories related to expences:
#     Alimony_Expenses, Pharmacy, ATM, Spending, Charity, Household, Decor, Delicatessen, Discounters, Child_Related, Electronics, Fast_Food, Hypermarket, Hobbies, Cafes, Cinema, Communication, Electricity, Fuel, Real_Estate, Renovation, Restaurant, Spa, Borrowing, School, Kindergarten, Theater, Telephone_and_TV_subscription, Flight_tickets, Train_tickets, Bus_tickets, OC/AC_insurance, Clothes, Medical_Services, Cleaning_services, Culture_Expenses
# 
# Categories related to incomes:
#     500 plus, Alimony_Income, Retirement, Rent, Scholarship, Receipts, Salary
# 
# All of single category consists of 7 type of variables: 
# 1. The total amount of related to category type customer expense/income in the last full month.
# 2. The total amount of related to category type customer expense/income in the last 3 months.
# 3. The total amount of related to category type customer expense/income in the last 6 months.
# 4. Average amount of related to category type customer expense/income in the last 3 months.
# 5. The total number of related to category type customer expense/income in the last full month.
# 6. The total number of related to category type customer expense/income in the last 3 month.
# 7. The total number of related to category type customer expense/income in the last 6 month.

# ## Correlations

# In[553]:


# Compute the correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 15, center="light", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, cbar_kws={"shrink": .5} )


# As mentioned previously the number of variables is to big to read from the heatmap many specific information. However, some information can by deducted like the fact that variables are only correlated with those from the same category, what can be better deducted from the heatmap of just first 3 categories of variables. 

# In[554]:


# Compute the correlation matrix
df_first3c = df.iloc[:, :22]
corr = df_first3c.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 15, center="light", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.01,
            square=True, cbar_kws={"shrink": .5})


# The presented correlation matrix confirms the hypothesis about the high correlation of variables within the category, and shows that no variables seem to be correlated with the searched variable Y. 
# 
# By the next step we can check which type of variables in general have bigger correlation to searched variable Y. This may allow to pick only a group of one type of variable to be analyzed as a representative group of a whole dataset.

# In[555]:


def avg_corr(dfc):
    dfc["Y"]=df["Y"]
    return dfc.corr().iloc[:,-1:].mean().values[0]

df_m_value = df.iloc[:,1::7]
df_3m_value = df.iloc[:,2::7]
df_6m_value = df.iloc[:,3::7]
df_avg = df.iloc[:,4::7]
df_m_number = df.iloc[:,5::7]
df_3m_number = df.iloc[:,6::7]
df_6m_number = df.iloc[:,7::7]

print(f'Average correlation for full month values of expense/income: {format(avg_corr(df_m_value),".4f")}.')
print(f'Average correlation for 3 months values of expense/income: {format(avg_corr(df_3m_value),".4f")}.')
print(f'Average correlation for 6 month values of expense/income: {format(avg_corr(df_6m_value),".4f")}.')
print(f'Average correlation for average values of expense/income: {format(avg_corr(df_avg),".4f")}.')
print(f'Average correlation for full month numbers of expense/income: {format(avg_corr(df_m_number),".4f")}.')
print(f'Average correlation for 3 month numbers of expense/income: {format(avg_corr(df_3m_number),".4f")}.')
print(f'Average correlation for 6 month numbers of expense/income: {format(avg_corr(df_6m_number),".4f")}.')


# Every type of variable has similar correlation with regards to the searched variable Y. However, to have a better insight to the data we will plot a correlation matrix based on Average correlation for full month numbers of expense/income as it achived the biggest value.

# In[556]:


# Compute the correlation matrix
corr = df_m_number.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 15, center="light", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.01,
            square=True, cbar_kws={"shrink": .5})


# From the above correlation matrix, we can point that variables X26 and X152 show above average correlation with variables from other groups. However, as one type of variable is not really helpfull to get a better insight to the dataset, we can try to choose variable within a category that has the biggest correlation to searched variable Y.

# In[557]:


# var = list of maximal correlated varaible within a category
def max_corr(dfc):
    dfc["Y"]=df["Y"]
    return abs(dfc.corr().iloc[:-1,-1:]).idxmax().values[0]
var = []
for i in range(1,295,7):
    df_c = df.iloc[:,i:i+7]
    var.append(max_corr(df_c))
    
print(f'Average maximal correlation within a category is {format(df.loc[:,var].corr().iloc[:,-1:].mean().values[0],".4f")} which is much better than in previous try. However, it is still very low correlation.')
    
corr = df.loc[:,var].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 15, center="light", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.01,
            square=True, cbar_kws={"shrink": .5})


# Lastly we can check which single variable is the most correlated to the searched varaible Y, which may be usuful in later analysis.

# In[558]:


# maximal correlated varaible
max_cor_var=abs(df.corr().iloc[1:,:1]).idxmax().values[0]
max_cor_val=df.corr().iloc[1:,:1].loc["X20"].values[0]
print(f"Maximal correlated variable is {max_cor_var} and the value of correlation to searched Y is {format(max_cor_val,'.4f')}.")


# ## Distribution

# Due to the fact that each variable has the majority of zero values, the classic box and distribution plots become completely unreadable. For this reason, the following graph will be shown without zero values

# In[570]:


def alter_dist(x):
    f, (ax_box, ax_dist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    right = 2.5*df[df[x]!=0][x].quantile(0.75) - 1.5*df[df[x]!=0][x].quantile(0.25)
    alt_df = df[(df[x]<right) & (df[x]>0)]
    
    sns.kdeplot(data=alt_df, x=x, hue="Y", bw=0.5, multiple="stack", palette="Set3", ax=ax_dist)
    sns.boxplot(df[df[x]>0][x], ax=ax_box, palette="Set3")
    text = f'There is {format((df[x]==0).sum(),".0f")} zero values. And {format((df[x]>right).sum(),".0f")} outliers '
    ax_box.set(xlabel=text)
    plt.xlim(0,)
    plt.show()
    
alter_dist("X8")


# In[560]:


def red_flag(a,b):
    if a>=2*b:
        return 1
    else:
        return 0
#df["Red_Flag1"] = df.apply(lambda x: red_flag(x["X290"], x["X289"]), axis = 1)
#df["Red_Flag2"] = df.apply(lambda x: red_flag(x["X283"], x["X282"]), axis = 1)
#df["Red_Flag3"] = df.apply(lambda x: 1 if x["Red_Flag"]==1 and x["Red_Flag2"]==1 else 0, axis = 1)
#df["Red_Flag"]=0 if: (df["X290"]-2*df["X289"])>=0


# In[561]:


# If 1 month 3 months and 6 months are not deflecting in more than 45%
def income_stability(a,b,c):
    if 3*a>0.55*b and 3*a<1.45*b and 2*b>0.55*c and 2*b<1.45*c:
        return 0
    else:
        return 1
df["Income_Unstability"] = df.apply(lambda x: income_stability(x["X281"], x["X282"],x["X283"]), axis = 1)

def salary_stability(a,b,c):
    if 3*a>0.55*b and 3*a<1.45*b and 2*b>0.55*c and 2*b<1.45*c:
        return 0
    else:
        return 1
df["Salary_Unstability"] = df.apply(lambda x: salary_stability(x["X288"], x["X289"],x["X290"]), axis = 1)


# In[562]:


with pd.option_context('display.max_columns', None):  # more options can be specified also
    print(df)


# In[563]:


df.corr()


# In[564]:


from sklearn.metrics import confusion_matrix, classification_report

print('Income_Unstability confussion matrix')
print(classification_report(df['Y'],df['Income_Unstability']))
print('Salary_Unstability confussion matrix')
print(classification_report(df['Y'],df['Salary_Unstability']))
'''
cf_matrix = confusion_matrix(df['Salary_Unstability'],df['Income_Unstability'])
ax = sns.heatmap(cf_matrix, annot=True)
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
'''


# In[565]:


df.describe()


# ## Transformation

# In[176]:


from scipy import stats
df['X10_log']= stats.boxcox(df['X10'])
#df['X10_log'] = np.log(df['X10']+0.01)
df['X10_log']

