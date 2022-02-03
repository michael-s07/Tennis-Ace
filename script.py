
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())

plt.scatter(df.BreakPointsOpportunities,df.Winnings)
plt.show()

features = df[['FirstServeReturnPointsWon']]
outcome = df.Winnings

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

reg = LinearRegression()
reg.fit(features_train, outcome_train)
first_score = reg.score(features_test, outcome_test)
print(first_score)
pred = reg.predict(features_test)

plt.scatter(outcome_test, pred, alpha=0.5)
plt.show()

#single feature linear regression 
f2=df.BreakPointsOpportunities
o2=df.Winnings

#spliting data set 
x1, x2, y1, y2 =train_test_split(f2, o2, train_size = 0.8)
x1= np.array(x1)
x1 = x1.reshape(-1, 1)
print(x1.shape)

x2= np.array(x2)
x2 = x2.reshape(-1, 1)

reg2 = LinearRegression()

reg2.fit(x1, y1)
print("The Score of predicting BreakPointsOpportunities and winnings is ", reg2.score(x2, y2))
y2= np.array(y2)
y2 = y2.reshape(-1, 1)

p2 = reg2.predict(y2)

plt.scatter(x2,p2 )
plt.show()

#Mulitple Regressing 
features = df[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
winnings = df[['Winnings']]

features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

print('Predicting Winnings with 2 Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.show()