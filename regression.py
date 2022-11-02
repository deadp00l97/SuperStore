import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the csv file
df = pd.read_csv("storefront.csv")
df = df[['OrderDate', 'Segment', 'Quantity']]

# Change OrderDate to Date type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df['OrderWeek'] = df['OrderDate'].dt.isocalendar().week

# create new dataframe
df = df[['OrderWeek', 'Segment', 'Quantity']]
df = df.groupby(['OrderWeek', 'Segment'])['Quantity'].sum().reset_index()

# replacing values
df['Segment'].replace(['Consumer', 'Corporate', 'Home Office'], [0, 1, 2], inplace=True)

# data splitting
X = df[['Segment', 'OrderWeek']]
y = df['Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# fit the model
classifier = LinearRegression()
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("regression.pkl", "wb"))