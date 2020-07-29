#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
print("python {}".format(sys.version))
print("python {}".format(scipy.__version__))
print("python {}".format(numpy.__version__))
print("python {}".format(matplotlib.__version__))
print("python {}".format(pandas.__version__))
print("python {}".format(sklearn.__version__))


# In[15]:


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[18]:


url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=["sepal-length","sepal-width","petal-length","petal-width","class"]
dataset = read_csv(url,names=names)


# In[19]:


print(dataset.shape)


# In[20]:


print(dataset.head(20))


# In[21]:


print(dataset.describe())


# In[22]:


print(dataset.groupby('class').size())


# In[24]:


dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()


# In[25]:


dataset.hist()
pyplot.show()


# In[26]:


scatter_matrix(dataset)
pyplot.show()


# In[27]:


array=dataset.values
x=array[:,0:4]
y=array[:, 4]
x_train,x_valudation,y_train,y_valudation=train_test_split(x,y,test_size=0.2,random_state=1)


# In[30]:


models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[31]:


results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f (%f)' %(name,cv_results.mean(),cv_results.std()))


# In[32]:


pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparsion')
pyplot.show()


# In[33]:


model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_valudation)


# In[35]:


print(accuracy_score(y_valudation,predictions))
print(confusion_matrix(y_valudation,predictions))
print(classification_report(y_valudation,predictions))


# In[ ]:




