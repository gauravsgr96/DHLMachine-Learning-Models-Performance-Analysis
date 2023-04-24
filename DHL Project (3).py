#!/usr/bin/env python
# coding: utf-8

# In[342]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt


# In[343]:


data = pd.read_csv(r"C:\Users\DELL\Desktop\BackOrders.csv")


# In[344]:


print(data)


# In[345]:


data.describe()


# In[346]:


data.isnull()


# In[347]:


data.info()


# In[348]:


data.isnull().sum()


# In[349]:


remove=['lead_time']
data.drop(remove,inplace=True,axis=1)


# In[350]:


data.duplicated()


# In[351]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\BackOrders.csv")


# In[352]:



df=df.groupby('lead_time')['in_transit_qty'].sum().to_frame().reset_index()


plt.barh(df['lead_time'],df['in_transit_qty'],color = ['#F0F8FF','#E6E6FA','#B0E0E6']) 

plt.title('Chart title')
plt.xlabel('X axis title')
plt.ylabel('Y axis title') 

plt.show()


# In[353]:



sns.barplot(x = 'lead_time',y = 'in_transit_qty',data =df,palette = "Blues")

plt.title('Chart title')
plt.xlabel('X axis title')
plt.ylabel('Y axis title') 
plt.xticks(rotation=90)

plt.show()


# In[354]:



plt.scatter(df['lead_time'],df['in_transit_qty'],alpha=0.5 )

plt.title('Chart title')
plt.xlabel('X axis title')
plt.ylabel('Y axis title') 

plt.show()


# In[355]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[356]:


data = pd.read_csv(r"C:\Users\DELL\Desktop\BackOrders.csv")


# In[357]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)
print(classification_report(y_test, y_pred))

# Plot ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load the data
data = pd.read_csv(r"C:\Users\DELL\Desktop\BackOrders.csv")

# Perform feature engineering and select relevant features
# ...

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance the data using undersampling or oversampling
# ...

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC()
}

# Train and evaluate the models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1_score:.4f}')
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})')
    
# Show the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




