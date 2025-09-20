import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

url = "FuelConsumptionCo2.csv"

df = pd.read_csv(url)

# take a look at the dataset
testdf = df.sample(5)
print(testdf)

# statistical summary of the data
print(df.describe())

# select features that might be indicative of CO2 emissions
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.sample(9))

# consider the histogram for each of the features
viz = df[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# checking scatter plot between fuel consumption and CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()

# checking scatter plot between engine size and CO2 emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.xlim(0, 27)
plt.show()

# checking scatter plot between number of cylinders and CO2 emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Number of Cylinders")       
plt.ylabel("CO2 Emissions")
plt.show()

#extracting the input features and target labels from datasets
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(type(X_train), np.shape(X_train))

from sklearn import linear_model
# create a linear regression object
regression = linear_model.LinearRegression()
# train the model using the training sets 
regression.fit(X_train.reshape(-1, 1), y_train) #convert x_train to 2D array

#print the coefficients
print('Coefficients: ', regression.coef_[0])
print('Intercept: ', regression.intercept_)

# visualize the model
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regression.predict(X_train.reshape(-1, 1)), color='red')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.savefig("regression_plot.png") #save the plot as a PNG file
plt.show()


# evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# use the predict() method to make test predictions
y_pred = regression.predict(X_test.reshape(-1, 1)) #convert x_test to 2D array

# print the evaluation metrics
print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score: %.2f" % r2_score(y_test, y_pred))
