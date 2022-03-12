import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

IceCream = pd.read_csv("data/IceCreamData.csv")
X = IceCream[['Temperature']]
y = IceCream[['Revenue']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression(fit_intercept=True)
model.fit(X_train,y_train)

#y_pred = model.predict(X_test)
#model.predict(np.array([[10]]))
    
X_test = np.array([[5]])
model.predict(X_test)

import pickle
pickle.dump(model, open("model-icecream.pkl","wb"))
model_pk = pickle.load(open("model-icecream.pkl","rb"))
model.predict(np.array([[40]]))