#regression type -output is item_outlet_sales
#we combine both train and test data into one.

import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10000)
#Read files:
test =pd.read_csv("H:\\data\\Test.csv")
train = pd.read_csv("H:\\data\\Train.csv")

train['source'] = 'Train'
test['source'] = 'Test'

data = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, data.shape)


data.apply(lambda x: sum(x.isnull())) #checking missing values.
print(data.apply(lambda x: sum(x.isnull())))

data.describe()
print(data.describe()) #getting some basic data.


data.apply(lambda x: len(x.unique()))
print(data.apply(lambda x: len(x.unique())))


#UNIVARIATE ANALYSIS.
train.info()
train.columns
sns.distplot(train.Item_Outlet_Sales, color= "r")
plt.show() #Item_Outlet_Sales is Positively Skewed

train.Item_Outlet_Sales.describe()
sns.distplot(train.Item_Visibility, color = "b")
plt.show()
#visibility is Higher for lot of items

sns.distplot(train.Item_Weight.dropna(), color = "y")
plt.show() #not any pattern for the item weight,



#Item Fat Content
train.Item_Fat_Content.value_counts().plot(kind = 'bar')
plt.show()


test.head()
test.Item_Fat_Content.value_counts().plot(kind = 'bar')
plt.show()




train.Item_Type.value_counts().plot(kind="bar")
plt.show()

#Outlet_Identifier
train.Outlet_Identifier.value_counts().plot(kind="bar")
plt.show() #less frequency count is out 10 and out 19

#Outlet_Size
train.Outlet_Size.value_counts().plot(kind="bar")
plt.show()

#Outlet Type
train.Outlet_Type.value_counts().plot(kind="bar")
plt.show()

# Outlet_Location_Type

train.Outlet_Location_Type.value_counts().plot(kind="bar")
plt.show()


#bivariate analysis


plt.figure(figsize = [8,8])
plt.scatter(train.Item_Visibility, train.Item_Outlet_Sales, color = "r")
plt.show()

#plt.figure(figsize = [10,8])
plt.scatter(train.Item_MRP, train.Item_Outlet_Sales, color = "b")
plt.show()

plt.figure(figsize = [10,8])
sns.violinplot(train.Item_Fat_Content, train.Item_Outlet_Sales)
plt.show()


sns.boxplot(train.Item_Fat_Content, train.Item_Outlet_Sales)
plt.show()

train.groupby("Item_Fat_Content")["Item_Outlet_Sales"].describe().T
plt.figure(figsize = [12,9])
sns.boxplot(train.Item_Type, train.Item_Outlet_Sales)
plt.xticks(rotation = 90)
plt.title("Boxplot - Item Type vs Sales")
plt.xlabel("Item Type")
plt.ylabel("Sales")
plt.show()



train.groupby("Item_Type")["Item_Outlet_Sales"].describe().T
plt.figure(figsize = [12,9])
sns.boxplot(train.Outlet_Identifier, train.Item_Outlet_Sales)
plt.xticks(rotation = 90)
plt.title("Boxplot - Outlet Identifier vs Sales")
plt.xlabel("Outlet Identifier")
plt.ylabel("Sales")
plt.show()

# From the boxplot we plotted at the beginning, we noticed that the item_weight column is approximately normal and it is therefore helpful to replace the missing values with the Mean of the column.
data['Item_Weight'].mean() #we will replace the NaN values with this mean

data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True) #missing values have been replaced with the mean using the fillna function.

#We will replace the missing values in Outlet_Size with the item that appears frequently, in this case Meduim.
data['Outlet_Size'].value_counts()
data['Outlet_Size'].fillna('Medium', inplace=True)

data.isnull().sum() #now we do not have any null values in Outlet_Size



data[data['Item_Visibility']==0]['Item_Visibility'].count()
data['Item_Visibility'].fillna(data['Item_Visibility'].median(), inplace=True)



data['Outlet_Establishment_Year'].value_counts()
data['Outlet_Years'] = 2009-data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


#Under normal circumstance, if a product is more visible, then it's likely it will be getting higher sales. We can based on that hypothesis and create importance given to a product in a given store according to the mean of significance given to the same product in all other stores.
item_visib_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
item_visib_avg
function = lambda x: x['Item_Visibility']/item_visib_avg['Item_Visibility'][item_visib_avg.index == x['Item_Identifier']][0]
data['item_visib_avg'] = data.apply(function,axis=1).astype(float)


data.head()






from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Label = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type']

for i in Label:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])
    
train.head()







y = train['Item_Outlet_Sales']
X_train = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
X_test = test.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()

from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize=True)

lr.fit(X_test,y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)





from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=100,n_jobs=4)

rf.fit(X_train,y)

rf_accuracy = round(rf.score(X_train,y)*100)

rf_accuracy





