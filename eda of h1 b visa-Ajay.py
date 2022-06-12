# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:56:39 2022

@author: LOST SOUL
"""

import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)



data =pd.read_csv("F:\\data major\\H-1B_Disclosure_Data_FY2019.csv")
data.head()
data.shape


data = data[:200000]
data.shape

pd.set_option('display.max_columns', None)
data.head()

data = data[data.VISA_CLASS == 'H-1B']
data.EMPLOYER_COUNTRY.value_counts()

data= data[data.EMPLOYER_COUNTRY == 'UNITED STATES OF AMERICA']

data.apply(lambda x:len(x.unique()))

data.isnull().sum()[data.isnull().sum() > 0]

to_select = ['CASE_NUMBER', 'CASE_STATUS', 'PERIOD_OF_EMPLOYMENT_START_DATE','EMPLOYER_NAME', 'EMPLOYER_STATE','JOB_TITLE',
             'SOC_TITLE','FULL_TIME_POSITION','PREVAILING_WAGE_1','PW_UNIT_OF_PAY_1','WORKSITE_STATE_1']



data1 = data[to_select]
data1.head()

emp = data1['EMPLOYER_NAME'].value_counts()[:10]
sns.barplot(x= emp.values, y = emp.index)


fig = plt.figure(figsize=(7,7))

fig.patch.set_facecolor('#F1FBFF')
fig.patch.set_alpha(1.0)

sums = data1.CASE_NUMBER.groupby(data.CASE_STATUS).count()
pie(sums, labels = sums.index, autopct='%.1f%%', textprops={'fontsize': 12})

ax1 = plt.title('Case Status of H-1B Visa',
          fontsize = 16,
          fontweight = 'heavy',
          loc = 'center', 
          pad = 30); #semi-colon for hide text before graph output




plt.rcParams["figure.figsize"] = (10,5)
(data1.CASE_STATUS.value_counts(normalize=True)*100).plot(kind='barh',title='H1B Petitions by Case Status')

top = data1.groupby('EMPLOYER_NAME').CASE_STATUS.count().nlargest(20).index.tolist()
top_data = data1.loc[data1.EMPLOYER_NAME.isin(top)]
top_data.groupby('SOC_TITLE').EMPLOYER_NAME.count().nlargest(10).plot(kind='barh',title='Occupation of the top h1b companies')


plt.figure(figsize=(10,8))
ax = data1['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10].plot.barh(width=0.9,color='#ffd700')
for i, v in enumerate(data['EMPLOYER_NAME'].value_counts().sort_values(ascending=False).values[:10]): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Highest Employeer')
fig=plt.gca()
fig.invert_yaxis()
plt.show()



plt.figure(figsize=(12,6))
data[data['PREVAILING_WAGE_1']<150000].PREVAILING_WAGE_1.hist(bins=40,color='khaki')
plt.axvline(data[data['PREVAILING_WAGE_1']<=150000].PREVAILING_WAGE_1.median(), color='green', linestyle='dashed', linewidth=4)
plt.title('Wage Distribution')
plt.show()



sns.countplot(data1['FULL_TIME_POSITION'])








