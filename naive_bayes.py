#-------------------------------------------------------------------------
# AUTHOR: Kate Yuan
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
one = ['Sunny', 'Hot', 'High', 'Strong']
two = ['Overcast', 'Mild', 'Normal', 'Weak']
three =['Rain', 'Cool']
rowLen = len(dbTraining[0])
for i in range(len(dbTraining)):
  X.append([])
  for j in range(rowLen - 1):
    if dbTraining[i][j] in one:
        X[i].append(1)
    elif dbTraining[i][j] in two:
        X[i].append(2)
    elif dbTraining[i][j] in three:
        X[i].append(3)
#print(X)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
values = ["Yes", "No"]
rowLen = len(dbTraining[0])
for i in range(len(dbTraining)):
  if dbTraining[i][rowLen - 1] == values[0]:
      Y.append(1)
  elif dbTraining[i][rowLen - 1] == values[1]:
      Y.append(2)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf = clf.fit(X, Y) #.predict(X_test)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())


testX = []
one = ['Sunny', 'Hot', 'High', 'Strong']
two = ['Overcast', 'Mild', 'Normal', 'Weak']
three =['Rain', 'Cool']
rowLen = len(dbTraining[0])
for i in range(len(dbTest)):
  testX.append([])
  for j in range(rowLen - 1):
    if dbTest[i][j] in one:
        testX[i].append(1)
    elif dbTest[i][j] in two:
        testX[i].append(2)
    elif dbTest[i][j] in three:
        testX[i].append(3)


#Printing the header of the solution
#--> add your Python code here
headers = df.columns.tolist()
for name in headers:
   print(f'{name:<12}', end = "")
print('Confidence')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
count = 0
for i in testX:
   prob = clf.predict_proba([i])[0]
   if(prob[0] > 0.75):
      row = df.loc[count]
      for val in row:
        if val == '?':
            print(f'{values[0]:<12}', end = "")
        else:
            print(f'{val:<12}', end = "")
      print(round(prob[0], 3))
   elif(prob[1] > 0.75):
      row = df.loc[count]
      for val in row:
        if val == '?':
            print(f'{values[1]:<12}', end = "")
        else:
            print(f'{val:<12}', end = "")
      print(round(prob[1], 3))
   count+= 1


