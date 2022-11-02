import pickle
import pandas as pd
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


# Load the csv file
df = pd.read_csv("data1.csv")
df = df[['OrderWeek', 'Segment', 'City', 'State', 'Region', 'SubCategory', 'Discount']].copy()
df.columns = ['OrderWeek', 'Segment', 'City', 'State', 'Region', 'SubCategory', 'Discount']


# Encode categorical data
le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])
df['OrderWeek'] = le.fit_transform(df['OrderWeek'])
df['Segment'] = le.fit_transform(df['Segment'])
df['State'] = le.fit_transform(df['State'])
df['Region'] = le.fit_transform(df['Region'])
df['SubCategory'] = le.fit_transform(df['SubCategory'])
df = df.astype({'Discount': 'category'})


# Split dataset
X = df.drop(['Discount'], axis=1)
y = df['Discount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# creating RF classifier
classifier = RandomForestClassifier(n_estimators=100)


# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
classifier.fit(X_train, y_train)


# Make pickle file of our model
pickle.dump(classifier, open("classification.pkl", "wb"))