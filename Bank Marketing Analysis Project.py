#!/usr/bin/env python
# coding: utf-8

# In[145]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[146]:


# To Avoid the Warning
# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[147]:


# seting to view entire dataset
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[148]:


# load data set
df=pd.read_csv('bank.csv')
df


# In[149]:


term_deposits = df.copy()
# Have a grasp of how our data looks.
df.head()


# In[150]:


# this dataset is used to display last few rowas
df.tail()


# In[151]:


df.isnull()


# In[152]:


# This function is used  to check the null values
df.isnull().sum()


# In[153]:


df.dropna(inplace=True)


# In[154]:


# to display the  categorical column
df.columns


# # 1.2 Gethering informations

# In[155]:


# getting size and shape
print(f"This data has {df.shape[0]} rows/enties and {df.shape[1]} columns/features.")


# In[156]:


df.info()


# In[157]:


# getting and sorting all the categorical variable
categorical_features = [feature for feature in df.columns if df[feature].dtype=='o']
print(f"Numbers of Categorical Features : {len(categorical_features)}")

# getting and sorting all the numerical  variable
numerical_features = [feature for feature in df.columns if df[feature].dtype!='o']
print(f"Numbers of Numerical Features : {len(numerical_features)}")


# In[158]:


df.describe().T


# In[159]:


df["deposit"].describe()


# In[160]:


# to display  the count of unique in deposit dataset
df["deposit"].value_counts()


# In[ ]:





# In[161]:


# ML  Libraries
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# In[162]:


# statical summary of all column
df.iloc[:, :-1].describe().T.sort_values(by='std', ascending = False)\
                          .style.background_gradient(cmap="Greens")\
                          .bar(subset=["max"], color="#F8766D")\
                          .bar(subset=["mean"], color="#00BFC4")


# In[163]:


# # 2. Data Visualization
# # 2.1 Outlier Analysis

plt.figure(figsize = (12,5))
sns.boxplot(df)
plt.grid()
plt.show()


# In[164]:


fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(221)
g = sns.boxplot(x="default", y="balance", hue="deposit", data=df, palette="muted", ax=ax1)
g.set_title("Amount of Balance by Term Suscriptions")

ax2 = fig.add_subplot(222)
# ax.set_xticklabels(df["default"].unique(), rotation=45, rotation_mode="anchor")
g1 = sns.boxplot(x="job", y="balance", hue="deposit", data=df, palette="RdBu", ax=ax2)
g1.set_xticklabels(df["job"].unique(), rotation=60, rotation_mode="anchor")
g1.set_title("Type of Work by Term Suscriptions")
plt.show()


# In[165]:


# violinplot for job distribution of balance by education
ax3 = fig.add_subplot(212)
# violinplot 
g2 = sns.violinplot(data=df, x="education", y="balance", hue="deposit") # palette="RdBu_r" color change
g2.set_title("Distribution of Balance by Education")
plt.show()


# In[166]:


# violinplot for job distribution of balance by deposite status

fig = plt.figure(figsize=(12,8))

sns.violinplot(x="balance", y="job", hue="deposit", data=df);

plt.show()


# In[167]:


f, ax = plt.subplots(1,2, figsize=(16,8))

labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"

plt.suptitle('Information on Term Suscriptions', fontsize=20)

df["deposit"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, startangle=25)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

# sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)
# ax[1].set_title('Condition of Loans', fontsize=20)
# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')

sns.barplot(x="education", y="balance", hue="deposit", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax[1].set(ylabel="(%)")
ax[1].set_xticklabels(df["education"].unique(), rotation=0, rotation_mode="anchor")
plt.show()


# In[168]:


# Let's see how the numeric data is distributed by histogram plot.

df.hist(bins=20, figsize=(14,10), color='#FA5858')

plt.show()

Marital Status:
Well in this analysis we didn't find any significant insights other than most divorced individuals are broke. No wonder since they have to split financial assets! Nevertheless, since no further insights have been found we will proceed to clustering marital status with education status. Let's see if we can find other groups of people in the sample population.
# In[169]:


df['marital'].value_counts()


# In[170]:


df['marital'].unique()


# In[171]:


df['marital'].value_counts().tolist()


# In[172]:


# bar plot for find out marital status
df1 = df['marital'].value_counts().tolist()
labels = ['married', 'divorced', 'single']
data = sns.barplot(x=labels,y=df1, color="#FE9A2E")
plt.title=("Count by Marital Status")
plt.show()


# In[173]:


# importing libries
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[174]:


# Distribution of Balances by Marital status
single = df['balance'].loc[df['marital'] == 'single'].values
married = df['balance'].loc[df['marital'] == 'married'].values
divorced = df['balance'].loc[df['marital'] == 'divorced'].values

single_dist = go.Histogram(x=single,histnorm='density', name='single',marker=dict(color='#6E6E6E'))
married_dist = go.Histogram(x=married,histnorm='density', name='married',marker=dict(color='#2E9AFE'))
divorced_dist = go.Histogram(x=divorced,histnorm='density', name='divorced',marker=dict(color='#FA5858'))

fig = tools.make_subplots(rows=3, print_grid=False)
fig.append_trace(single_dist, 1, 1)
fig.append_trace(married_dist, 2, 1)
fig.append_trace(divorced_dist, 3, 1)

fig['layout'].update(showlegend=False, title="Price Distributions by Marital Status",height=1000, width=800)
iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[175]:


df = df.drop(df.loc[df["education"] == "unknown"].index)
df['education'].unique()


# In[176]:


df['marital/education'] = np.nan
lst = [df]

for col in lst:
    col.loc[(col['marital'] == 'single') & (df['education'] == 'primary'), 'marital/education'] = 'single/primary'
    col.loc[(col['marital'] == 'married') & (df['education'] == 'primary'), 'marital/education'] = 'married/primary'
    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'primary'), 'marital/education'] = 'divorced/primary'
    col.loc[(col['marital'] == 'single') & (df['education'] == 'secondary'), 'marital/education'] = 'single/secondary'
    col.loc[(col['marital'] == 'married') & (df['education'] == 'secondary'), 'marital/education'] = 'married/secondary'
    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'secondary'), 'marital/education'] = 'divorced/secondary'
    col.loc[(col['marital'] == 'single') & (df['education'] == 'tertiary'), 'marital/education'] = 'single/tertiary'
    col.loc[(col['marital'] == 'married') & (df['education'] == 'tertiary'), 'marital/education'] = 'married/tertiary'
    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'tertiary'), 'marital/education'] = 'divorced/tertiary'
    
    
df.head()


# In[177]:


# bar plot for marital/education and balance
education_groups = df.groupby(['marital/education'], as_index=False)['balance'].median()
fig = plt.figure(figsize=(12,8))

sns.barplot(x="balance", y="marital/education", data=education_groups,label="Total",palette="RdBu")
plt.show()


# # 2.2 Multivariate Analysis
# a) Pair Plot Based on Marital Status

# In[87]:


# list of feature used for pair plot
sns.set(style="ticks")

# create a pairplot
sns.pairplot(df, hue="marital/education", palette="plasma")
plt.show()


# # 3.1 Heat Map
# a) correlational matric

# In[ ]:





# In[178]:


# drop marital/education and balance status
# scale both numeric and categorical vaues
# Then let's use a correlation matrix
# With that we can determine if duration has influence on term deposits
#from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

fig = plt.figure(figsize=(15,5))
df['deposit'] = LabelEncoder().fit_transform(df['deposit'])

# Separate both dataframes into 
numeric_df = df.select_dtypes(exclude="object")

# categorical_df = df.select_dtypes(include="object")
corr_numeric = numeric_df.corr()

sns.heatmap(corr_numeric, annot=True, cmap="coolwarm")

plt.show()


# In[179]:


numeric_df.corr()


# In[180]:


dep = term_deposits['deposit']
term_deposits.drop(labels=['deposit'], axis=1,inplace=True)
term_deposits.insert(0, 'deposit', dep)
term_deposits.head()
# housing has a -20% correlation with deposit let's see how it is distributed.
# 52 %
term_deposits["housing"].value_counts()/len(term_deposits)


# In[181]:


term_deposits["loan"].value_counts()/len(term_deposits)


# In[182]:


# Here we split the data into training and test sets and implement a stratified shuffle split.
stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_set, test_set in stratified.split(term_deposits, term_deposits["loan"]):
    stratified_train = term_deposits.loc[train_set]
    stratified_test = term_deposits.loc[test_set]
    
stratified_train["loan"].value_counts()/len(df)
stratified_test["loan"].value_counts()/len(df)


# In[183]:


# Separate the labels and the features.
train_data = stratified_train # Make a copy of the stratified training set.
test_data = stratified_test
train_data.shape
test_data.shape
train_data['deposit'].value_counts()


# In[184]:


from sklearn.model_selection import train_test_split


# In[185]:


X = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
y = df['deposit']


# In[186]:


# Assuming X and y are already defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[187]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[188]:


#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
y = df['deposit']
m = LinearRegression()
m.fit(X_train,y_train)


# In[189]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assuming you have already defined X_train and y_train
# Assuming the columns 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous' are present in your dataset
X = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
y = df['deposit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Linear Regression
m = LinearRegression()
m.fit(X_train, y_train_encoded)


# In[190]:


df_str = df.to_string()
print(df_str)


# In[191]:


m.score(X_test,y_test)


# In[192]:


output = m.predict(X_test)
output


# In[193]:


y_test


# In[194]:


compaire = pd.DataFrame({'actual_value':y_test,'predict_value':output})
compaire


# In[195]:


# Assuming compaire DataFrame is already created
plt.scatter(compaire['actual_value'], compaire['predict_value'])
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.show()


# In[196]:


from sklearn import metrics

mean_aberror = metrics.mean_absolute_error(y_test,output)
mean_sqerror = metrics.mean_squared_error(y_test,output)
rmsqurrerror = np.sqrt(metrics.mean_squared_error(y_test,output))
print(m.score(X,y)*100)
print(mean_aberror) #0.00000000000004298186828178065
print(mean_sqerror)
print(rmsqurrerror)


# In[197]:


#k-nearest neighbors

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train,y_train)
knn_predictions=knn_model.predict(X_test)
#mean squared error
knn_mse=mean_squared_error(y_test,knn_predictions)
print("means squared error:",knn_mse)


# In[198]:


# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Assume you have your feature matrix X and target variable y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_reg_model = LinearRegression()

# Train the model on the training data
linear_reg_model.fit(X_train, y_train)

# Make predictions on the test data
linear_reg_predictions = linear_reg_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, linear_reg_predictions)
print("Mean Squared Error:", mse)

# Print the coefficients and intercept
print("Coefficients:", linear_reg_model.coef_)
print("Intercept:", linear_reg_model.intercept_)


# In[199]:


# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))


# In[200]:


# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))


# In[201]:


# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_pred))


# In[202]:


#GradientBoosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))


# In[203]:


# Accuracy scores
accuracy_scores = {
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "Random Forest": accuracy_score(y_test, rf_pred),
    "Decision Tree": accuracy_score(y_test, dt_pred),
    "Gradient Boosting": accuracy_score(y_test, gb_pred)
}


# In[205]:


# Create bar plot
plt.figure(figsize=(8, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green', 'red', 'yellow'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


# In[ ]:





# In[207]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Assume you have your feature matrix X and target variable y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
print("KNN Mean Squared Error:", knn_mse)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print("Random Forest Mean Squared Error:", rf_mse)


# In[208]:


from sklearn.metrics import mean_squared_error

# Assuming knn_model and rf_model are already trained and tested
knn_predictions = knn_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

knn_mse = mean_squared_error(y_test, knn_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

print("KNN Mean Squared Error:", knn_mse)
print("Random Forest Mean Squared Error:", rf_mse)


# In[209]:


from sklearn.metrics import r2_score

knn_r2 = r2_score(y_test, knn_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("KNN R-squared:", knn_r2)
print("Random Forest R-squared:", rf_r2)


# In[210]:


# Access feature importances for Random Forest
feature_importances = rf_model.feature_importances_
print("Random Forest Feature Importances:", feature_importances)


# In[214]:


from sklearn.model_selection import cross_val_score

# Cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='neg_mean_squared_error')
print("KNN Cross-Validation Mean MSE:", np.mean(-knn_cv_scores))

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Random Forest Cross-Validation Mean MSE:", np.mean(-rf_cv_scores))


# In[ ]:




