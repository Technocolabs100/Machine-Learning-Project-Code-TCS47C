#!/usr/bin/env python
# coding: utf-8

# # **EXPLORATORY DATA ANALYSIS OF H1-B VISA DATASET 2019 :**
# 
# 
# ###  IMPORTING LIBRARIES ##

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[137]:


import numpy as np 
import re
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly
import getpass 
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from statistics import mode
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)


# In[2]:


df = pd.read_csv('H-1B_Disclosure_Data_FY2019(csv).csv', low_memory = False)
df.head()


# In[3]:


df.shape


# In[4]:


# df = df[:200000]
# df.shape


# In[5]:


pd.set_option('display.max_columns', None)
df.head()


# In[6]:


df.apply(lambda x:len(x.unique()))


# In[7]:


df = df[df.VISA_CLASS == 'H-1B']


# In[ ]:


df.CASE_STATUS.value_counts()


# In[8]:


warnings.filterwarnings("ignore")
df.CASE_STATUS[df['CASE_STATUS']=='WITHDRAWN'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS']=='CERTIFIED-WITHDRAWN'] = 'CERTIFIED'


# In[8]:


df.EMPLOYER_COUNTRY.value_counts()


# In[9]:


df= df[df.EMPLOYER_COUNTRY == 'UNITED STATES OF AMERICA']


# In[10]:


df.isnull().sum()[df.isnull().sum() > 0]


# ###  SELECTING RELEVANT FEATURES -

# In[83]:


to_select = ['CASE_NUMBER', 'CASE_STATUS', 'EMPLOYER_NAME', 'SECONDARY_ENTITY_1', 'AGENT_REPRESENTING_EMPLOYER',
             'PERIOD_OF_EMPLOYMENT_START_DATE', 'JOB_TITLE', 'SOC_TITLE', 'SOC_CODE', 'NAICS_CODE', 'FULL_TIME_POSITION',
             'NEW_CONCURRENT_EMPLOYMENT', 'PREVAILING_WAGE_1', 'CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT', 
             'CHANGE_EMPLOYER', 'AMENDED_PETITION', 'H-1B_DEPENDENT', 'SUPPORT_H1B', 'WILLFUL_VIOLATOR',
             'WAGE_RATE_OF_PAY_FROM_1', 'WAGE_UNIT_OF_PAY_1', 'TOTAL_WORKER_POSITIONS']

to_select_1 = ['WAGE_RATE_OF_PAY_FROM_1', 'WAGE_UNIT_OF_PAY_1', 'PREVAILING_WAGE_1', 'WAGE_RATE_OF_PAY_TO_1']


# In[84]:


df1 = df[to_select]
df1.head()

df2 = df[to_select_1]


# In[85]:


df1.dtypes


# ### DISTRIBUTION OF VISA CASES WITH THE COMPANIES -

# In[14]:


emp = df1['EMPLOYER_NAME'].value_counts()[:10]
sns.barplot(x= emp.values, y = emp.index)


# ### * PIE chart showing different cases of CASE-STATUS :

# In[15]:


fig = plt.figure(figsize=(7,7))

fig.patch.set_facecolor('#F1FBFF')
fig.patch.set_alpha(1.0)

sums = df1.CASE_NUMBER.groupby(df1.CASE_STATUS).count()
pie(sums, labels = sums.index, autopct='%.1f%%', textprops={'fontsize': 12})

ax1 = plt.title('Case Status of H-1B Visa',
          fontsize = 16,
          fontweight = 'heavy',
          loc = 'center', 
          pad = 30); #semi-colon for hide text before graph output


# In[16]:


plt.rcParams["figure.figsize"] = (10,5)
(df1.CASE_STATUS.value_counts(normalize=True)*100).plot(kind='barh',title='H1B Petitions by Case Status')


# ###  What are the top OCCUPATIONS of the H1-B's being filed by the employers ?

# In[17]:


top = df1.groupby('EMPLOYER_NAME').CASE_STATUS.count().nlargest(20).index.tolist()
top_df = df1.loc[df1.EMPLOYER_NAME.isin(top)]
top_df.groupby('SOC_TITLE').EMPLOYER_NAME.count().nlargest(10).plot(kind='barh',title='Occupation of the top h1b companies')


# ### Which employers file the most petitions ?

# In[18]:


plt.figure(figsize=(10,8))
ax = df1['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10].plot.barh(width=0.9,color='#ffd700')
for i, v in enumerate(df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False).values[:10]): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Highest Employeer')
fig=plt.gca()
fig.invert_yaxis()
plt.show()


# ###  WAGE DISTRIBUTION -

# In[19]:


plt.figure(figsize=(12,6))
df1[df1['PREVAILING_WAGE_1']<150000].PREVAILING_WAGE_1.hist(bins=40,color='khaki')
plt.axvline(df[df['PREVAILING_WAGE_1']<=150000].PREVAILING_WAGE_1.median(), color='green', linestyle='dashed', linewidth=4)
plt.title('Wage Distribution')
plt.show()


# In[20]:


plt.figure(figsize=(12,7))
sns.set(style="whitegrid")
g = sns.countplot(x = 'FULL_TIME_POSITION', data = df1)
plt.title("NUMBER OF APPLICATIONS MADE FOR THE FULL TIME POSITION")
plt.ylabel("NUMBER OF PETITIONS MADE")
plt.show()


# In[21]:


cor = df1.corr()

plt.figure(figsize=(12,6))
sns.heatmap(cor, annot = True)


# In[22]:


df1.isnull().sum()


# # **FEATURE ENGINEERING**

# In[86]:


df1['CASE_STATUS'] = df1['CASE_STATUS'].map({'CERTIFIED' : 1, 'DENIED' : 0})


# In[87]:


df1.head()


# In[14]:


df1.AGENT_REPRESENTING_EMPLOYER.value_counts(dropna = False)


# In[15]:


df1.WILLFUL_VIOLATOR.value_counts(dropna = False)


# In[88]:


df1['FULL_TIME_POSITION'] = df1['FULL_TIME_POSITION'].map({'N' : 0, 'Y' : 1})
df1['AGENT_REPRESENTING_EMPLOYER'] = df1['AGENT_REPRESENTING_EMPLOYER'].map({'N' : 0, 'Y' : 1})
df1['SECONDARY_ENTITY_1'] = df1['SECONDARY_ENTITY_1'].map({'N' : 0, 'Y' : 1})
df1['H-1B_DEPENDENT'] = df1['H-1B_DEPENDENT'].map({'N' : 0, 'Y' : 1})
df1['WILLFUL_VIOLATOR'] = df1['WILLFUL_VIOLATOR'].map({'N' : 0, 'Y' : 1})
df1['SUPPORT_H1B'] = df1['SUPPORT_H1B'].map({'N' : 0, 'Y' : 1})


# In[17]:


df1.CONTINUED_EMPLOYMENT.value_counts(dropna = False)


# In[89]:


df1 = df1[df1.CONTINUED_EMPLOYMENT != 'B']
df1 = df1[df1.CONTINUED_EMPLOYMENT != '01']
df1 = df1[df1.CONTINUED_EMPLOYMENT != '00'] 
df1 = df1[df1.CONTINUED_EMPLOYMENT != '001'] 
df1 = df1[df1.CONTINUED_EMPLOYMENT != '02']


# In[90]:


df1['CONTINUED_EMPLOYMENT'] = df1['CONTINUED_EMPLOYMENT'].astype(float)


# In[91]:


df1['AGENT_REPRESENTING_EMPLOYER'] = df1['AGENT_REPRESENTING_EMPLOYER'].fillna(df1['AGENT_REPRESENTING_EMPLOYER'].mode()[0])
df1['SECONDARY_ENTITY_1'] = df1['SECONDARY_ENTITY_1'].fillna(df1['SECONDARY_ENTITY_1'].mode()[0])
df1['H-1B_DEPENDENT'] = df1['H-1B_DEPENDENT'].fillna(df1['H-1B_DEPENDENT'].mode()[0])
df1['WILLFUL_VIOLATOR'] = df1['WILLFUL_VIOLATOR'].fillna(df1['WILLFUL_VIOLATOR'].mode()[0])
df1['EMPLOYER_NAME'] = df1['EMPLOYER_NAME'].fillna(df1['EMPLOYER_NAME'].mode()[0])
df1['JOB_TITLE'] = df1['JOB_TITLE'].fillna(df1['JOB_TITLE'].mode()[0])
df1['SOC_CODE'] = df1['SOC_CODE'].fillna(df1['SOC_CODE'].mode()[0])
df1['NAICS_CODE'] = df1['NAICS_CODE'].fillna(df1['NAICS_CODE'].mode()[0])
df1['SOC_CODE'] = df1['SOC_CODE'].fillna(df1['SOC_CODE'].mode()[0])
df1['NEW_CONCURRENT_EMPLOYMENT'] = df1['NEW_CONCURRENT_EMPLOYMENT'].fillna(df1['NEW_CONCURRENT_EMPLOYMENT'].mode()[0])
df1['WAGE_UNIT_OF_PAY_1'] = df1['WAGE_UNIT_OF_PAY_1'].fillna(df1['WAGE_UNIT_OF_PAY_1'].mode()[0])
df1['TOTAL_WORKER_POSITIONS'] = df1['TOTAL_WORKER_POSITIONS'].fillna(df1['TOTAL_WORKER_POSITIONS'].mode()[0])


# In[92]:


df1.isnull().sum()


# In[21]:


df1['H-1B_DEPENDENT'].value_counts()


# In[93]:


df1['NEW_CONCURRENT_EMP'] = df1['NEW_CONCURRENT_EMPLOYMENT']
df1['NEW_CONCURRENT_EMP'] = np.where(df1['NEW_CONCURRENT_EMP'].isin([0]), '0',
                             np.where(df1['NEW_CONCURRENT_EMP'].isin([1]), '1', '>1'))

df1['CHANGE_PREVIOUS_EMP'] = df1['CHANGE_PREVIOUS_EMPLOYMENT']
df1['CHANGE_PREVIOUS_EMP'] = np.where(df1['CHANGE_PREVIOUS_EMPLOYMENT'].isin([0]), '0',
                             np.where(df1['CHANGE_PREVIOUS_EMPLOYMENT'].isin([1]), '1', '>1'))

# df1['CONTINUED_EMPLOYMENT_BIN'] = df1['CONTINUED_EMPLOYMENT']
# df1['CONTINUED_EMPLOYMENT_BIN'] = np.where(df1['CONTINUED_EMPLOYMENT'].isin([0]), '0',
#                              np.where(df1['CONTINUED_EMPLOYMENT'].isin([1]), '1', '>1'))

df1['AMENDED_PETITION_BIN'] = df1['AMENDED_PETITION']
df1['AMENDED_PETITION_BIN'] = np.where(df1['AMENDED_PETITION'].isin([0]), '0',
                             np.where(df1['AMENDED_PETITION'].isin([1]), '1', '>1'))

df1['CHANGE_EMPLOYER_BIN'] = df1['CHANGE_EMPLOYER']
df1['CHANGE_EMPLOYER_BIN'] = np.where(df1['CHANGE_EMPLOYER'].isin([0]), '0',
                             np.where(df1['CHANGE_EMPLOYER'].isin([1]), '1', '>1'))

df1['TOTAL_WORKER_POSITIONS_BIN'] = df1['TOTAL_WORKER_POSITIONS']
df1['TOTAL_WORKER_POSITIONS_BIN'] = np.where(df1['TOTAL_WORKER_POSITIONS'].isin([0]), '0',
                             np.where(df1['TOTAL_WORKER_POSITIONS'].isin([1]), '1', '>1'))


# In[94]:


df1.TOTAL_WORKER_POSITIONS_BIN.value_counts()


# In[95]:


df1.AMENDED_PETITION_BIN.value_counts()


# In[96]:


df1['PREVAILING_WAGE_1'].value_counts(dropna = False)


# In[97]:


q1 = df1["PREVAILING_WAGE_1"].quantile(0.25)
q3 = df1["PREVAILING_WAGE_1"].quantile(0.75)
IQR = q3 - q1
mean = df1['PREVAILING_WAGE_1'].mean()


# In[98]:


df1['PREVAILING_WAGE_1'] = df1['PREVAILING_WAGE_1'].apply(lambda x: x if x != None and (x <= q1 + 1.5*IQR and x >= q1 - 1.5*IQR) else mean)


# In[28]:


plt.figure(figsize=(20,20))
sns.displot(df1['PREVAILING_WAGE_1'], bins = 50)


# In[29]:


df1.boxplot(column='PREVAILING_WAGE_1')


# In[30]:


df1.isnull().sum()


# In[31]:


df1.SOC_TITLE.value_counts()


# In[99]:


df1['OCCUPATION'] = np.nan
df1['SOC_TITLE'] = df1['SOC_TITLE'].str.lower()
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('computer','programmer', na=False)] = 'Computer Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('software','web developer', na=False)] = 'Computer Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('database', na=False)] = 'Computer Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('math','statistic', na=False)] = 'Mathematical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('predictive model','stats', na=False)] = 'Mathematical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('teacher','linguist', na=False)] = 'Education Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('professor','Teach', na=False)] = 'Education Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('school principal', na=False)] = 'Education Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('medical','doctor', na=False)] = 'Medical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('physician','dentist', na=False)] = 'Medical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('Health','Physical Therapists', na=False)] = 'Medical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('surgeon','nurse', na=False)] = 'Medical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('psychiatry', na=False)] = 'Medical Occupations'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('chemist','physicist', na=False)] = 'Advance Sciences'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('biology','scientist', na=False)] = 'Advance Sciences'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('biologi','clinical research', na=False)] = 'Advance Sciences'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('public relation','manage', na=False)] = 'Management Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('management','operation', na=False)] = 'Management Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('chief','plan', na=False)] = 'Management Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('executive', na=False)] = 'Management Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('advertis','marketing', na=False)] = 'Marketing Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('promotion','market research', na=False)] = 'Marketing Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('business','business analyst', na=False)] = 'Business Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('business systems analyst', na=False)] = 'Business Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('accountant','finance', na=False)] = 'Financial Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('financial', na=False)] = 'Financial Occupation'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('engineer','architect', na=False)] = 'Architecture & Engineering'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('surveyor','carto', na=False)] = 'Architecture & Engineering'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('technician','drafter', na=False)] = 'Architecture & Engineering'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('information security','information tech', na=False)] = 'Architecture & Engineering'
df1.OCCUPATION[df1['SOC_TITLE'].str.contains('education','law', na=False)] = 'Administrative Occupation'

df1['OCCUPATION']= df1.OCCUPATION.replace(np.nan, 'Others', regex=True)


# In[33]:


df1.OCCUPATION.value_counts(dropna = False)


# In[100]:


df1['JOB_TITLE_NEW'] = 'others'
df1['JOB_TITLE'] = df1['JOB_TITLE'].str.upper()
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('IOS|DEVOPS|CLOUD|FRONT END|INTERIOR|.NET|DEVOPS|SOFTWARE|COMPUTER|INFORMATION|SECURITY|SYSTEMS|AUTOMATION|SYSTEMS|FULL STACK|LEAD|JAVA|IT|TEST|GRAPHIC|SUPPORT')] = 'IT & SOFTWARE ENGINEERS'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('QA|ENGAGEMENT|OPERATIONS|DELIVERY|INFRASTRUCTURE|FIRMWARE|ANDRIOD|UX|RF|PYTHON|TABLEAU|HADOOP|INFORMATICA|SQL|BI|SCRUM|VALIDATION|APPLICATIONS|UI|PROGRAMMER|DEVELOPER|SOLUTION|RPA')] = 'IT & SOFTWARE ENGINEERS'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('LANDSCAPE|CAD|SITE|FIELD|QUALITY|MECHANICAL DESIGN|STRUCTURAL|DESIGNER|SIMULATION|ENGINEERING|MARINE|INDUSTRIAL|MATERIALS|MECHANICAL|MANUFACTURING|CIVIL')] = 'MECHANICAL & CIVIL ENGINEER '
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('ACCOUNTANT|FINANCIAL|QUANTITATIVE|RISK|BUDGET|TAX')] = 'FINANCE TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('PRESIDENT|DIRECTOR|MANAGER')] = 'Manager & DIRECTORS'
#H1B_visa['JOB_TITLE_NEW'][H1B_visa['JOB_TITLE'].str.contains('ELECTRICAL|CHEMICAL')] = 'ELECTRICAL ENGINEERS'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('SERVICE|AEM|EMBEDDED|DIGITAL|NETWORK|CONTROLS|HARDWARE|FUNCTIONAL|ELECTRICAL|CHEMICAL')] = 'ELECTRONICS & ELECTRONICS ENGINEERS TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('PUBLIC|LAWYERS|ATTORNEY|LAW')] = 'LAW TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('SALESFORCE|MARKET|MARKETING|SUPPLY')] = 'MARKETING TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('SPEECH|BIG|ORACLE|MACHINE|DATABASE|DATA|SCIENTIST|ASSOCIATES')] = 'DATABASE & SCIENTISTS'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('ARCHITECT|ARCHITECTURAL')] = 'ARCHITECT'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('TEACHER|PROFESSOR|POSTDOCTORAL|FELLOW|SCHOLAR|LECTURER|LABORATORY')] = 'EDUCATIONAL ORGANISATION'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('BUSINESS|ADMINISTRATOR|INVESTMENT|ACCOUNT')] = 'BUSINESS TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('DENTIST|HOSPITALIST|THERAPIST|PSYCHIATRIST|PEDIATRICIAN|PHYSICIAN|FAMILY|NEPHROLOGIST')] = 'MEDICAL TEAM'
df1['JOB_TITLE_NEW'][df1['JOB_TITLE'].str.contains('SENIOR|SR.|SR')] = 'SENIOR TEAM'


# In[47]:


df1.JOB_TITLE_NEW.value_counts(dropna = False)


# In[48]:


df1.head()


# In[49]:


df1.EMPLOYER_NAME.value_counts(dropna = False)


# In[101]:


# df1['NEW_EMPLOYER'] = np.nan
# df1.shape

# df1['EMPLOYER_NAME'] = df1['EMPLOYER_NAME'].str.lower()
# df1.NEW_EMPLOYER[df1['EMPLOYER_NAME'].str.contains('university', na = False)] = 'university'
# df1['NEW_EMPLOYER']= df1.NEW_EMPLOYER.replace(np.nan, 'non university', regex=True)

# df1['NEW_EMPLOYER'] = df1['NEW_EMPLOYER'].map({'university' : 1, 'non university' : 0})

df1['EMPLOYER_BRANCH'] = 'others'
df1['EMPLOYER_NAME'] = df1['EMPLOYER_NAME'].str.upper()
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('APPLE|GOOGLE|FACEBOOK|CAPGEMINI|WIPRO|TWITTER|INFOSYS|MICROSOFT|AIRLINES|IBM|ERNST|JPMORGAN|MINDTREE|AMAZON|TATA')] = 'TOP TECH'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('ELECTRONIC|MARIX|MICRO|ELECTRO|CHIP|DEVICE|INSTRUMENTS|INTEGRATORS|DELL|HEW|SEMICONDUCTORS|ENTERTAINMENT|LOGIC')] = 'ELECTRONIC & LOGISTICS SERVICES'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('UNIVERSITY|UNIVERSITIES|ACADEMIC|INSTITUTIONS|SCIENCE|NATIONAL|SCHOOL')] = 'UNIVERSITY'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('MASTER|BANK|CARD|VISA')] = 'BANKING COMPANIES'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('HEALTH|FIN|ECLINICALWORKS|MEDTRONIC|FINANCIAL|MEDICAL|MED|CENTER')] = 'FINANCE AND MEDICAL SOLUTIONS'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('BUSINESS|MANAGEMENT')] = 'BUSINESS SOLUTIONS'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('LABS|COMMUNICATION|NETWORK|DIGITAL|NETWORKS')] = 'RESEARCH LABS & NETWORK'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('AUTOBILE|AUTOMOTIVE|MOTOR|AUTO|FORD|PUMP|ELECTRIC|TESLA|BOSCH')] = 'AUTOMOTIVE & ELECTRICAL'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('DEVELOPMENT|IT|COMPUTER|CYBER|TECHNOLOGY|TECH|SOLUTIONS|WEB|INFOTECH|CLOUD|VISION|GLOBAL|SYSTEMS|TECHNOSOFT|TECHNO|SERVICES|SECURITIES|SECURITY|TECHNOLOGIES|DATA')] = 'TECH SOLUTIONS'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('INTERNATIONAL|CONSULTING|CONSULTANT|RESOURCES|GROUP|ASSOCIATES|ANALYSTS')] = 'CONSULTING COMPANIES'
df1['EMPLOYER_BRANCH'][df1['EMPLOYER_NAME'].str.contains('PRODUCT|PRODUCTS|ENTERPRISE|ENTERPRISES')] = 'PRODUCT &ENTERPRISE COMPANIES'


# In[75]:


df1.EMPLOYER_BRANCH.value_counts(dropna = False)


# In[76]:


df1.isnull().sum()


# In[102]:


df1 = df1.drop(['CASE_NUMBER', 'SOC_TITLE', 'EMPLOYER_NAME', 'PERIOD_OF_EMPLOYMENT_START_DATE', 'NEW_CONCURRENT_EMPLOYMENT', 'JOB_TITLE'
               , 'SOC_CODE', 'NAICS_CODE', 'CHANGE_PREVIOUS_EMPLOYMENT', 'AMENDED_PETITION', 'CHANGE_EMPLOYER'
               , 'TOTAL_WORKER_POSITIONS', 'PREVAILING_WAGE_1', 'WAGE_UNIT_OF_PAY_1'], axis = 1)


# In[103]:


df1.head()


# In[104]:


import seaborn as sns
plt.figure(figsize=(18, 14))
cor = df1.corr()
sns.heatmap(cor, annot = True, cmap = plt.cm.CMRmap_r)
plt.show()


# In[105]:


df1.dtypes


# In[106]:


df1['WAGE_RATE_OF_PAY_FROM_1'].value_counts(dropna = False)


# In[107]:


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# In[108]:


df1 = df1[df1['WAGE_RATE_OF_PAY_FROM_1'].apply(lambda x: isfloat(x))]


# In[109]:


# df1['WAGE_RATE_OF_PAY_FROM_1'] = df1['WAGE_RATE_OF_PAY_FROM_1'].str.replace("[^0-9]", "x", regex=True)


# In[110]:


df1['WAGE_RATE_OF_PAY_FROM_1'] = df1['WAGE_RATE_OF_PAY_FROM_1'].astype(float)


# In[111]:


df1 = df1.dropna(subset=['WAGE_RATE_OF_PAY_FROM_1'])


# In[112]:


df1['WAGE_RATE_OF_PAY_FROM_1'].isnull().sum()


# In[64]:


plt.figure(figsize=(20,20))
sns.displot(df1['WAGE_RATE_OF_PAY_FROM_1'], bins = 50)


# In[126]:


df1 = df1.drop(['SUPPORT_H1B'], axis = 1)


# In[127]:


df1.shape


# In[128]:


df1[['CASE_STATUS', 'SECONDARY_ENTITY_1', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H-1B_DEPENDENT', 
     'WILLFUL_VIOLATOR', 'NEW_CONCURRENT_EMP', 'CHANGE_PREVIOUS_EMP', 'AMENDED_PETITION_BIN', 
     'CHANGE_EMPLOYER_BIN', 'TOTAL_WORKER_POSITIONS_BIN', 'OCCUPATION', 'JOB_TITLE_NEW', 'EMPLOYER_BRANCH']] = df1[['CASE_STATUS', 'SECONDARY_ENTITY_1', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION','H-1B_DEPENDENT', 
     'WILLFUL_VIOLATOR', 'NEW_CONCURRENT_EMP', 'CHANGE_PREVIOUS_EMP', 'AMENDED_PETITION_BIN', 
     'CHANGE_EMPLOYER_BIN', 'TOTAL_WORKER_POSITIONS_BIN', 'OCCUPATION', 'JOB_TITLE_NEW', 'EMPLOYER_BRANCH']] .apply(lambda x: x.astype('category'))


# In[129]:


df1.dtypes


# In[130]:


y = df1.CASE_STATUS
X = df1.drop('CASE_STATUS', axis = 1)

seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train.columns


# In[131]:


X_train.isnull().sum()


# In[132]:


X_train_encode = pd.get_dummies(X_train)
X_test_encode = pd.get_dummies(X_test)


# In[133]:


X_test_encode.head()


# In[56]:


y_train.head()


# # **MODEL BUILDING**

# ### XGBCLASSIFIER

# In[ ]:


xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_encode, y_train)
y_pred = xgb_model.predict(X_test_encode.to_numpy())


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# ### LOGISTIC REGRESSION

# In[134]:


LogReg = LogisticRegression()
LogReg.fit(X_train_encode, y_train)
y_pred = LogReg.predict(X_test_encode.to_numpy())


# In[135]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[136]:


metrics.accuracy_score(y_test, y_pred)


# **LOGISTIC REGRESSION WITH CLASS - WEIGHTS**

# In[ ]:


LogReg = LogisticRegression(solver='newton-cg', class_weight='balanced')
LogReg.fit(X_train_encode, y_train)
y_pred = LogReg.predict(X_test_encode.to_numpy())


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# ### RANDOM FOREST

# In[ ]:


clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X_train_encode, y_train)
y_pred = clf.predict(X_test_encode)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# ### DOWNSAMPLING THE DATA

# In[ ]:


df2 = pd.concat([X_train_encode, y_train], axis=1)

df2_majority = df2[df2.CASE_STATUS==1]
df2_minority = df2[df2.CASE_STATUS==0]
 
# Downsample majority class
df2_majority_downsampled = resample(df2_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df2_minority),  # to match minority class
                                 random_state=1234) # reproducible results
 
# Combine minority class with downsampled majority class
df2_downsampled = pd.concat([df2_majority_downsampled, df2_minority])
 
# Display new class counts
df2_downsampled.head()


# In[ ]:


y = df2_downsampled.CASE_STATUS
X = df2_downsampled.drop('CASE_STATUS', axis = 1)


# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X_test_encode.to_numpy())


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# ### OVERSAMPLING THE DATA

# In[ ]:


df2 = pd.concat([X_train_encode, y_train], axis=1)

# Separate majority and minority classes
# Separate majority and minority classes
df2_majority = df2[df2.CASE_STATUS==1]
df2_minority = df2[df2.CASE_STATUS==0]

# Upsample minority class
df2_minority_upsampled = resample(df2_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df2_majority),    # to match majority class
                                 random_state=1234) # reproducible results
 
# Combine majority class with upsampled minority class
df2_upsampled = pd.concat([df2_majority, df2_minority_upsampled])


# In[ ]:


df2_upsampled.shape


# In[ ]:


y = df2_upsampled.CASE_STATUS
X = df2_upsampled.drop('CASE_STATUS', axis = 1)


# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X_test_encode.to_numpy())


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# ### OVERSAMPLING USING SMOTE

# In[57]:


from imblearn.over_sampling import SMOTE


# In[58]:


sm = SMOTE(sampling_strategy = 0.1)
X, y = sm.fit_resample(X_train_encode, y_train)


# In[ ]:


# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.7)
# fit and apply the transform
X1, y1 = under.fit_resample(X, y)


# In[ ]:


from sklearn.decomposition import PCA
 
pca = PCA(n_components = 2)
 
X1 = pca.fit_transform(X1)
X_test_encode = pca.transform(X_test_encode)


# **LOGISTIC REGRESSION**

# In[59]:


LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X_test_encode)


# In[60]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[61]:


metrics.accuracy_score(y_test, y_pred)


# **RANDOM FOREST**

# In[ ]:


clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X1, y1)
y_pred = clf.predict(X_test_encode)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


metrics.accuracy_score(y_test, y_pred)


# In[ ]:





# In[140]:


### Create a Pickle file using serialization 
import pickle
pickle_out = open("LogReg.pkl", "wb")
pickle.dump(LogReg, pickle_out)
pickle_out.close()


# In[ ]:




