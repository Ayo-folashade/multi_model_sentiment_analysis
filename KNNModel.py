# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Loading the dataset
data = pd.read_csv('Twitter_Data.csv')

# Dealing with missing values
data['clean_text'].fillna('', inplace=True)
data['category'].fillna(0, inplace=True)

# Splitting the dataset into train and test
X = data['clean_text']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extracting the features from the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training with the knn_classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_vectors, y_train)

# Making predictions on the test data
y_pred = knn_classifier.predict(X_test_vectors)

# Evaluating the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print(classification_report(y_test, y_pred))
