#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('creditcard.csv')

# Descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Pair plot
sns.pairplot(df)
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Pair plot
sns.pairplot(df)
plt.show()


# In[ ]:


from sklearn.impute import SimpleImputer

# Handling missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
df['column_with_nan'] = imputer.fit_transform(df[['column_with_nan']])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handling outliers using IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['column'] < (Q1 - 1.5 * IQR)) | (df['column'] > (Q3 + 1.5 * IQR)))]

# Normalize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean['scaled_column'] = scaler.fit_transform(df_clean[['column']])

# Verify cleaned data
df_clean.head()


# In[ ]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
print(Counter(df_clean['target']))

# Apply SMOTE to balance the classes
X = df_clean.drop('target', axis=1)
y = df_clean['target']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check new class distribution
print(Counter(y_resampled))


# In[ ]:


# Create new features (date extraction)
df_clean['date_column'] = pd.to_datetime(df_clean['date_column'])
df_clean['year'] = df_clean['date_column'].dt.year
df_clean['month'] = df_clean['date_column'].dt.month
df_clean['day'] = df_clean['date_column'].dt.day

# One-hot encoding for categorical columns
df_encoded = pd.get_dummies(df_clean, columns=['categorical_column'])

# Feature scaling for numeric data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('target', axis=1))

# Verify feature-engineered data
df_encoded.head()


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# RandomForest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print('Cross-Validation Scores:', cv_scores)
print('Mean CV Score:', cv_scores.mean())


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_
print('Best Parameters:', grid_search.best_params_)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Predictions on test set
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# ROC AUC Score
y_prob = best_model.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_prob)
print('ROC AUC Score:', roc_score)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[ ]:


import pickle

# Save the model using Pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# To load the model
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions with the loaded model
new_predictions = loaded_model.predict(X_test)


# In[ ]:




