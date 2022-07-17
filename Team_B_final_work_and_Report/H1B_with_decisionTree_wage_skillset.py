#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# In[2]:


dataset = pd.read_csv('C:\my files\Machine_learning_internship\Main_project\H-1B_Disclosure_Data_FY2019.csv',low_memory=False)


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# # Selecting the features which will contribute prominently to the model building
#

# In[5]:


dataset = dataset[['CASE_STATUS','VISA_CLASS','EMPLOYER_NAME','AGENT_REPRESENTING_EMPLOYER','SECONDARY_ENTITY_1','JOB_TITLE','SOC_TITLE','SOC_CODE', 'NAICS_CODE','CONTINUED_EMPLOYMENT', 'CHANGE_PREVIOUS_EMPLOYMENT','NEW_CONCURRENT_EMPLOYMENT', 'CHANGE_EMPLOYER','AMENDED_PETITION', 'H-1B_DEPENDENT', 'SUPPORT_H1B','WILLFUL_VIOLATOR','WAGE_RATE_OF_PAY_FROM_1', 'WAGE_UNIT_OF_PAY_1','TOTAL_WORKER_POSITIONS','PREVAILING_WAGE_1','WAGE_RATE_OF_PAY_TO_1']]


# # selecting only those data points for which the visa type is H1 - B

# In[6]:


dataset.CASE_STATUS.unique()


# In[7]:


dataset = dataset[((dataset['CASE_STATUS'].str.upper() == 'CERTIFIED') |                                (dataset['CASE_STATUS'].str.upper() == 'DENIED')) &                               (dataset['VISA_CLASS'].str.upper() == 'H-1B')]


# In[1]:


dataset.CASE_STATUS.unique()


# # EDA

# In[9]:


dataset.CASE_STATUS.value_counts().plot(kind = 'bar')


# In[10]:


emp_name = dataset['EMPLOYER_NAME'].value_counts()
emp_name.to_frame()


# In[11]:


dataset['EMPLOYER_NAME'].value_counts()
sns.barplot(x = dataset['EMPLOYER_NAME'].value_counts()[:10], y = dataset['EMPLOYER_NAME'].value_counts().index[:10])


# In[12]:


sns.countplot(dataset.WILLFUL_VIOLATOR)


# In[13]:


dataset.AGENT_REPRESENTING_EMPLOYER.unique()


# In[14]:


sns.countplot(dataset.AGENT_REPRESENTING_EMPLOYER)


# In[15]:


dataset.JOB_TITLE.unique()


# In[16]:


dataset.head()


# In[17]:


dataset.info()


# In[18]:


dataset['H-1B_DEPENDENT'].head()


# In[19]:


dataset['H-1B_DEPENDENT'].unique()


# In[20]:


sns.countplot(dataset['H-1B_DEPENDENT'])


# In[21]:


ammendedPetition = dataset.AMENDED_PETITION
ammendedPetition.to_frame()


# In[22]:


dataset.CHANGE_EMPLOYER.head(30)


# In[23]:


dataset.CHANGE_EMPLOYER.unique()


# In[24]:


sns.countplot(dataset.CHANGE_EMPLOYER)


# In[25]:


dataset['SECONDARY_ENTITY_1'].head(50)


# In[26]:


dataset['SECONDARY_ENTITY_1'].unique()


# In[27]:


dataset['SECONDARY_ENTITY_1'].value_counts()


# In[28]:


sns.countplot(dataset['SECONDARY_ENTITY_1'])


# # Dealing with missing values

# In[29]:


dataset.isnull().sum()


# In[30]:


dataset.SUPPORT_H1B.value_counts()


# In[31]:


dataset.SUPPORT_H1B.describe()


# In[32]:


dataset.SUPPORT_H1B.fillna("Y",inplace = True)


# In[33]:


dataset.isnull().sum()


# In[34]:


dataset['SECONDARY_ENTITY_1'].describe()


# In[35]:


dataset['SECONDARY_ENTITY_1'].fillna("N",inplace = True)


# In[36]:


dataset.isnull().sum()


# In[37]:


dataset.SUPPORT_H1B.describe()


# In[38]:


dataset.SUPPORT_H1B.unique()


# In[39]:


sns.countplot(dataset.SUPPORT_H1B)


# In[40]:


dataset.SUPPORT_H1B.fillna("Y",inplace = True)


# In[41]:


sns.boxplot(dataset['WAGE_RATE_OF_PAY_FROM_1'])


# In[42]:


dataset['WAGE_RATE_OF_PAY_FROM_1'].describe()


# In[43]:


dataset['WAGE_RATE_OF_PAY_FROM_1'].isnull().sum()


# In[44]:


#sns.boxplot(dataset['WAGE_UNIT_OF_PAY_1'])
sns.countplot(dataset['WAGE_UNIT_OF_PAY_1'])


# In[45]:


def convert(x):
  if x=='Y':
    return(0)
  else:
    return(1)


# In[46]:


dataset['TOTAL_WORKER_POSITIONS'].value_counts()


# In[47]:


sns.boxplot(dataset['PREVAILING_WAGE_1'])


# In[48]:


plt.scatter(dataset['PREVAILING_WAGE_1'], dataset['CASE_STATUS'])


# In[49]:


dataset.pivot_table('PREVAILING_WAGE_1','CASE_STATUS').plot.bar()


# In[50]:


dataset['PREVAILING_WAGE_1'] = dataset['PREVAILING_WAGE_1'].clip(lower = dataset['PREVAILING_WAGE_1'].quantile(0.1), upper = dataset['PREVAILING_WAGE_1'].quantile(0.80))


# In[51]:


dataset['WAGE_RATE_OF_PAY_FROM_1'] = dataset['WAGE_RATE_OF_PAY_FROM_1'].clip(lower = dataset['WAGE_RATE_OF_PAY_FROM_1'].quantile(0.1), upper = dataset['WAGE_RATE_OF_PAY_FROM_1'].quantile(0.80))


# In[52]:


sns.boxplot(dataset['WAGE_RATE_OF_PAY_FROM_1'])


# In[53]:


dataset['WAGE_RATE_OF_PAY_FROM_1'].isnull().sum()


# In[54]:


sns.boxplot(dataset['PREVAILING_WAGE_1'])


# In[55]:


dataset['PREVAILING_WAGE_1'].count()


# In[56]:


dataset['PREVAILING_WAGE_1'].isnull().value_counts()


# In[57]:


dataset['PREVAILING_WAGE_1'].describe()


# In[58]:


dataset['PREVAILING_WAGE_1'] = dataset['PREVAILING_WAGE_1'].fillna(dataset['PREVAILING_WAGE_1'].mean())


# In[59]:


dataset['PREVAILING_WAGE_1'].isnull().value_counts()


# # dropping the null values of EMPLOYER_NAME,                  AGENT_REPRESENTING_EMPLOYER, SOC_TITLE, SOC_CODE,                  NAICS_CODE

# In[60]:



dataset.dropna(axis = 0, inplace =True)


# In[61]:


dataset.isnull().sum()


# # transforming the Y and N into 0 and 1

# In[62]:


def transformation(x):
  if x=='Y':
    return(0)
  else:
    return(1)


# In[63]:


dataset['SUPPORT_H1B'] = dataset['SUPPORT_H1B'].apply(transformation)


# In[64]:


dataset.SUPPORT_H1B.value_counts()


# In[65]:


dataset['SECONDARY_ENTITY_1'] = dataset['SECONDARY_ENTITY_1'].apply(transformation)


# In[66]:


dataset['SECONDARY_ENTITY_1'].value_counts()


# In[67]:


dataset['H-1B_DEPENDENT'] = dataset['H-1B_DEPENDENT'].apply(transformation)


# In[68]:


dataset['H-1B_DEPENDENT'].unique()


# In[69]:


dataset['WILLFUL_VIOLATOR'] = dataset['WILLFUL_VIOLATOR'].apply(transformation)


# In[70]:


dataset['WILLFUL_VIOLATOR'].unique()


# In[71]:


dataset['WILLFUL_VIOLATOR'].value_counts()


# In[72]:


dataset['AGENT_REPRESENTING_EMPLOYER'].unique()


# In[73]:


dataset['AGENT_REPRESENTING_EMPLOYER'] = dataset['AGENT_REPRESENTING_EMPLOYER'].apply(transformation)


# In[74]:


dataset['AGENT_REPRESENTING_EMPLOYER'].value_counts()


# # checking with the JOB_TITLE,	SOC_TITLE,	SOC_CODE,	NAICS_CODE features

# In[75]:


dataset['JOB_TITLE'].value_counts()
sns.barplot(x = dataset['JOB_TITLE'].value_counts()[:10], y = dataset['JOB_TITLE'].value_counts().index[:10])


# In[76]:


dataset['JOB_TITLE'].nunique()


# In[77]:


dataset['SOC_TITLE'].value_counts()
sns.barplot(x = dataset['SOC_TITLE'].value_counts()[:10], y = dataset['SOC_TITLE'].value_counts().index[:10])


# In[78]:


dataset['SOC_CODE'].value_counts()
sns.barplot(x = dataset['SOC_CODE'].value_counts()[:10], y = dataset['SOC_CODE'].value_counts().index[:10])


# In[79]:


dataset['NAICS_CODE'].value_counts()


# In[80]:


dataset['NAICS_CODE'].nunique()


# In[81]:


dataset['SOC_CODE'].nunique()


# In[82]:


#dataset['JOB_TITLE'] = dataset['JOB_TITLE'].str.replace("%" , "").astype(float)
#dataset.loc[dataset['JOB_TITLE']=='ADVERSTING AND PROMOTIONS MANAGER']


# In[83]:


#dataset['SUPPORT_H1B'].mode().values[0]


# # using label encoding to convert string data points of a variable into numeric data

# In[84]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(dataset['CASE_STATUS'])
label


# In[85]:


dataset.drop("CASE_STATUS", axis=1, inplace=True)
dataset["CASE_STATUS"] = label
dataset["CASE_STATUS"].head(30)


# In[86]:


import sys
dataset['SOC_TITLE_NEW'] = 'others'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('WEB|SOFTWARE|COMPUTER|INFORMATION|SECURITY')] = 'IT ENGINEERS'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('MECHANICAL|DESIGN')] = 'MECHANICAL'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('CHIEF|EXECUTIVES')] = 'Executives'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('Chief|MANAGEMENT|MANAGERS')] = 'Manager'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('CHEMICAL|MARINE|INDUSTRIAL|MATERIALS')] = 'MULTIDISCPLINARY ENGINEERS'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('DATA|Database|STATISTICIANS')] = 'Database & Scientists'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('Sales|Market')] = 'Sales & Market'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('FINANCIAL|ECONOMISTS')] = 'Finance'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('COMPLIANCE|PUBLIC RELATIONS|Fundraising')] = 'P.R'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('education|law')] = 'Administrative'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('ACCOUNTANTS|Auditors|Compliance')] = 'Audit'
#H1B_visa['SOC_TITLE_NEW'][H1B_visa['SOC_TITLE'].str.contains('Recruiters|HUMAN RESOURCES|')] = 'H.R'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('Agricultural|Farm')] = 'Agriculture'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('Construction|Architectural')] = 'Estate'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('INTERNISTS|DENTISTS|THERAPISTS|SURGEONS|BIOMEDICAL')] = 'Medical'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('WRITERS|TEACHERS|POSTSECONDARY|KINDERGARTEN AND ELEMENTARY SCHOOL')] = 'Education'
dataset['SOC_TITLE_NEW'][dataset['SOC_TITLE'].str.contains('TECHNICIANS|WORKERS|CHEMISTS|BIOCHEMISTS')] = 'TECHNICIANS'


# In[87]:


dataset.head()



# In[110]:


import sys
dataset['JOB_TITLE_NEW'] = 'others'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('IOS|DEVOPS|CLOUD|FRONT END|INTERIOR|.NET|DEVOPS|SOFTWARE|COMPUTER|INFORMATION|SECURITY|SYSTEMS|AUTOMATION|SYSTEMS|FULL STACK|LEAD|JAVA|IT|TEST|GRAPHIC|SUPPORT')] = 'IT & SOFTWARE ENGINEERS'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('QA|ENGAGEMENT|OPERATIONS|DELIVERY|INFRASTRUCTURE|FIRMWARE|ANDRIOD|UX|RF|PYTHON|TABLEAU|HADOOP|INFORMATICA|SQL|BI|SCRUM|VALIDATION|APPLICATIONS|UI|PROGRAMMER|DEVELOPER|SOLUTION|RPA')] = 'IT & SOFTWARE ENGINEERS'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('LANDSCAPE|CAD|SITE|FIELD|QUALITY|MECHANICAL DESIGN|STRUCTURAL|DESIGNER|SIMULATION|ENGINEERING|MARINE|INDUSTRIAL|MATERIALS|MECHANICAL|MANUFACTURING|CIVIL')] = 'MECHANICAL & CIVIL ENGINEER '
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('ACCOUNTANT|FINANCIAL|QUANTITATIVE|RISK|BUDGET|TAX')] = 'FINANCE TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('PRESIDENT|DIRECTOR|MANAGER')] = 'Manager & DIRECTORS'
#H1B_visa['JOB_TITLE_NEW'][H1B_visa['JOB_TITLE'].str.contains('ELECTRICAL|CHEMICAL')] = 'ELECTRICAL ENGINEERS'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('SERVICE|AEM|EMBEDDED|DIGITAL|NETWORK|CONTROLS|HARDWARE|FUNCTIONAL|ELECTRICAL|CHEMICAL')] = 'ELECTRONICS & ELECTRONICS ENGINEERS TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('PUBLIC|LAWYERS|ATTORNEY|LAW')] = 'LAW TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('SALESFORCE|MARKET|MARKETING|SUPPLY')] = 'MARKETING TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('SPEECH|BIG|ORACLE|MACHINE|DATABASE|DATA|SCIENTIST|ASSOCIATES')] = 'DATABASE & SCIENTISTS'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('ARCHITECT|ARCHITECTURAL')] = 'ARCHITECT'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('TEACHER|PROFESSOR|POSTDOCTORAL|FELLOW|SCHOLAR|LECTURER|LABORATORY')] = 'EDUCATIONAL ORGANISATION'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('BUSINESS|ADMINISTRATOR|INVESTMENT|ACCOUNT')] = 'BUSINESS TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('DENTIST|HOSPITALIST|THERAPIST|PSYCHIATRIST|PEDIATRICIAN|PHYSICIAN|FAMILY|NEPHROLOGIST')] = 'MEDICAL TEAM'
dataset['JOB_TITLE_NEW'][dataset['JOB_TITLE'].str.contains('SENIOR|SR.|SR')] = 'SENIOR TEAM'


# In[91]:


dataset['SOC_CODE'] = dataset['SOC_CODE'].replace(['OPERATIONS RESEARCH ANALYSTS'],'15')
dataset['SOC_CODE_NEW'] = dataset['SOC_CODE'].str.split("-").str[0]


# In[92]:



dataset['SOC_CODE_NEW'].unique()


# In[93]:


dataset['SOC_CODE'].unique()

dataset['SOC_CODE'] = dataset['SOC_CODE'].replace(['OPERATIONS RESEARCH ANALYSTS'],'15')
dataset['SOC_CODE_NEW'] = dataset['SOC_CODE'].str.split("-").str[0]
dataset['SOC_CODE_NEW'] = dataset['SOC_CODE_NEW'].replace(['39','35','53','51','47','49','31','33','45','37'],'1000')#'10 codes lessthan 100')
dataset['SOC_CODE_NEW'].value_counts()

# In[94]:


dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['001','01'],'1')
dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['00'],'0')
dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['02'],'2')
dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['B'],'1')
dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['0',],'1')
dataset['CONTINUED_EMPLOYMENT'] = dataset['CONTINUED_EMPLOYMENT'].replace(['25','20','15','6','8','12','30','50','40','18','35','13','7','99','45','17','21','11'],'lower values 100 frequency')
dataset['CONTINUED_EMPLOYMENT'].value_counts()


# In[95]:


dataset['NAICS_CODE_NEW'] = dataset['NAICS_CODE'].astype(str).str[0:2]
dataset['NAICS_CODE_NEW'].value_counts()


# In[132]:


dataset['NAICS_CODE'].value_counts().unique()


# In[96]:


import sys
dataset['EMPLOYER_BRANCH'] = 'others'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('APPLE|GOOGLE|FACEBOOK|CAPGEMINI|WIPRO|TWITTER|INFOSYS|MICROSOFT|AIRLINES|IBM|ERNST|JPMORGAN|MINDTREE|AMAZON|TATA')] = 'TOP TECH'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('ELECTRONIC|MARIX|MICRO|ELECTRO|CHIP|DEVICE|INSTRUMENTS|INTEGRATORS|DELL|HEW|SEMICONDUCTORS|ENTERTAINMENT|LOGIC')] = 'ELECTRONIC & LOGISTICS SERVICES'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('UNIVERSITY|UNIVERSITIES|ACADEMIC|INSTITUTIONS|SCIENCE|NATIONAL|SCHOOL')] = 'UNIVERSITY'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('MASTER|BANK|CARD|VISA')] = 'BANKING COMPANIES'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('HEALTH|FIN|ECLINICALWORKS|MEDTRONIC|FINANCIAL|MEDICAL|MED|CENTER')] = 'FINANCE AND MEDICAL SOLUTIONS'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('BUSINESS|MANAGEMENT')] = 'BUSINESS SOLUTIONS'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('LABS|COMMUNICATION|NETWORK|DIGITAL|NETWORKS')] = 'RESEARCH LABS & NETWORK'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('AUTOBILE|AUTOMOTIVE|MOTOR|AUTO|FORD|PUMP|ELECTRIC|TESLA|BOSCH')] = 'AUTOMOTIVE & ELECTRICAL'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('DEVELOPMENT|IT|COMPUTER|CYBER|TECHNOLOGY|TECH|SOLUTIONS|WEB|INFOTECH|CLOUD|VISION|GLOBAL|SYSTEMS|TECHNOSOFT|TECHNO|SERVICES|SECURITIES|SECURITY|TECHNOLOGIES|DATA')] = 'TECH SOLUTIONS'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('INTERNATIONAL|CONSULTING|CONSULTANT|RESOURCES|GROUP|ASSOCIATES|ANALYSTS')] = 'CONSULTING COMPANIES'
dataset['EMPLOYER_BRANCH'][dataset['EMPLOYER_NAME'].str.contains('PRODUCT|PRODUCTS|ENTERPRISE|ENTERPRISES')] = 'PRODUCT &ENTERPRISE COMPANIES'


# In[97]:


dataset.head()


#
#
# dataset.drop('EMPLOYER_BRANCH', axis=1, inplace=True)
# dataset.drop('SOC_TITLE_NEW', axis=1, inplace=True)
# dataset.drop('JOB_TITLE_NEW', axis=1, inplace=True)
#
# dataset.drop('CONTINUED_EMPLOYMENT', axis=1, inplace=True)
# dataset.drop('SOC_CODE_NEW', axis=1, inplace=True)
# dataset.drop('NAICS_CODE_NEW', axis=1, inplace=True)

# In[98]:


label1 = le.fit_transform(dataset['SOC_TITLE_NEW'])
dataset.drop("SOC_TITLE_NEW", axis=1, inplace=True)
dataset["SOC_TITLE_NEW"] = label1
dataset["SOC_TITLE_NEW"].head(30)


# In[99]:


dataset['SOC_TITLE_NEW'].unique()


# In[100]:


#label2 = le.fit_transform(dataset['NAICS_CODE_NEW'])
#dataset.drop("NAICS_CODE_NEW", axis=1, inplace=True)
#dataset["NAICS_CODE_NEW"] = label2
#dataset["NAICS_CODE_NEW"].head(30)'''


# In[101]:


dataset["NAICS_CODE_NEW"].head()


# In[109]:


dataset.head()


# In[103]:


label3 = le.fit_transform(dataset['EMPLOYER_BRANCH'])
dataset.drop("EMPLOYER_BRANCH", axis=1, inplace=True)
dataset["EMPLOYER_BRANCH"] = label3
dataset["EMPLOYER_BRANCH"].head(30)


# In[111]:


label5 = le.fit_transform(dataset['JOB_TITLE_NEW'])
dataset.drop("JOB_TITLE_NEW", axis=1, inplace=True)
dataset["JOB_TITLE_NEW"] = label5
dataset["JOB_TITLE_NEW"].head(30)


# In[112]:


dataset.head()


# In[113]:


dataset = dataset.drop('EMPLOYER_NAME', axis = 1)
dataset = dataset.drop('SOC_TITLE', axis = 1)
#dataset = dataset.drop('SOC_CODE', axis = 1)
#dataset = dataset.drop('JOB_TITLE', axis = 1)
#dataset = dataset.drop('NAICS_CODE', axis = 1)


# In[114]:


label6 = le.fit_transform(dataset['WAGE_UNIT_OF_PAY_1'])
dataset.drop('WAGE_UNIT_OF_PAY_1', axis=1, inplace=True)
dataset['WAGE_UNIT_OF_PAY_1'] = label6
dataset['WAGE_UNIT_OF_PAY_1'].head(30)


# In[115]:


dataset.corr()


# In[116]:


sns.heatmap(dataset.corr())


# In[117]:


dataset['WAGE_RATE_OF_PAY_TO_1'].head()


# In[ ]:





# # segregating the x and y features

# In[118]:


#x_features = dataset[['SOC_TITLE_NEW','EMPLOYER_BRANCH','JOB_TITLE_NEW','SOC_CODE_NEW','NAICS_CODE_NEW','WAGE_UNIT_OF_PAY_1','WAGE_RATE_OF_PAY_TO_1','WAGE_RATE_OF_PAY_FROM_1','PREVAILING_WAGE_1']]


# In[133]:


x_features = dataset[['SOC_TITLE_NEW','EMPLOYER_BRANCH','JOB_TITLE_NEW','SOC_CODE_NEW','WAGE_UNIT_OF_PAY_1','WAGE_RATE_OF_PAY_TO_1','WAGE_RATE_OF_PAY_FROM_1','PREVAILING_WAGE_1']]


# In[134]:


y_features = dataset[['CASE_STATUS']]


# In[135]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_features, y_features, test_size = 0.3, random_state = 0)


# In[136]:


#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


# # modelling part

# In[137]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# # we will use the decision trees but before that we will find the maximum depth of the tree

# In[138]:


accuracy = []
for i in range(1,10):
    model =  DecisionTreeClassifier(max_depth = i, random_state = 0)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test,pred)
    accuracy.append(score)


# In[139]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')


# # Training

# In[140]:


dtree = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5, random_state = 0)
dtree.fit(X_train,y_train)
dtree_predictions = dtree.predict(X_test)
accuracy = accuracy_score(y_test,dtree_predictions)
print('accuracy before resampling',accuracy)


# # dealing with imbalanced dataset

# In[141]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[142]:


accuracy1 = []
for i in range(1,20):
    model1 =  DecisionTreeClassifier(max_depth = i, random_state = 0)
    model1.fit(X_train_res,y_train_res)
    pred1 = model1.predict(X_test)
    score1 = accuracy_score(y_test,pred1)
    accuracy1.append(score1)


# In[143]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), accuracy1, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')


# In[144]:


dtree1 = DecisionTreeClassifier(criterion = 'entropy',max_depth = 19 , random_state = 0)
dtree1.fit(X_train_res,y_train_res)
dtree_predictions1 = dtree1.predict(X_test)
#accuracy_score = accuracy_score(y_test,dtree_predictions1)
print("Accuracy of the Model after resampling: {0}%".format(accuracy_score(y_test, dtree_predictions1)*100))


# In[ ]:

filename =  'trained_model_H1B_visa_wage_skillset.pkl'
pickle.dump(model , open(filename, 'wb'))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
