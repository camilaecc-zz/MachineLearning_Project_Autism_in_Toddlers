
import pandas as panda

''' import data visualization libraries '''
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


data = panda.read_csv('/Users/Camila/Desktop/AI_Autism_Project /Toddler_Autism_dataset.csv')
print(data.head())

''' Modifying data and removing columns I wont be using '''
d = data.drop(['Who completed the test'], axis = 1, inplace = True)
d = data.replace({"yes": 1, "no": 0,"Yes": 1, "No": 0, "f": 1, "m": 0}).convert_dtypes(int)
dt = data.replace({"yes": 1, "no": 0,"Yes": 1, "No": 0, "f": 1, "m": 0}).convert_dtypes(int)
print(d.head())

'''Heat Map Correlation'''
plt.figure(figsize=(15,15))
cor = d.corr().abs()
cor_target = abs(cor["Class/ASD Traits "])
sns.heatmap(data = cor, cmap=plt.cm.Blues, annot = True, square = True, cbar = True )

print("\nFrom highest to lowest correlation\n")
print(cor_target[cor_target>-1].sort_values(ascending=False))


print(plt.show())

X = d.iloc[:,:-2].values
y = dt['Class/ASD Traits ']

'''Supervised learning'''
print("Supervised learning")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("\nTrain Data: ")
print(X_train.shape, y_train.shape)
print("\nTest Data: ")
print(X_test.shape, y_test.shape)
print()

'''Testing with 'Target' dropped '''
d.drop(['Class/ASD Traits ','Ethnicity'], axis = 1, inplace=True)
X = d.iloc[:,:-2].values

print("Testing with 'Target' dropped ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("\nTrain Data: ")
print(X_train.shape, y_train.shape)
print("\nTest Data: ")
print(X_test.shape, y_test.shape)
print()

'''Regression and classification techniques '''
y= y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Regression and classification techniques")
models = []
models.append(('Linear Regression  :', LinearRegression()))
models.append(('Logistic Regression:', LogisticRegression()))
models.append(('Naive Bayes        :', GaussianNB()))
models.append(('Lasso              :', Lasso(alpha=0.1)))
models.append(('Stochastic Gradient Descent:', SGDClassifier(loss="log", penalty="l2", shuffle=True, max_iter=100) ))

for name, model in models:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test).astype(int)
    print(name, accuracy_score(y_test, prediction))

'''Predictions and Actual'''
df = panda.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': prediction.flatten()})
print("\nAcutal and Prediction")
with panda.option_context('display.max_rows', 20, 'display.max_columns', None):
    print(df)

'''Accuracy Score'''
score = round(accuracy_score(prediction,y_test)*100,2)
print("\nAccuracy Score of training and testing")
print("Score: " + str(score) + "%")