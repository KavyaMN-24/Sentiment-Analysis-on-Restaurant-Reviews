from google.colab import drive
drive.mount('/content/drive/')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def predict_sentiment(sample_review, classifier, cv):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    temp = cv.transform([final_review]).toarray()
    prediction = classifier.predict(temp)

    return prediction


# Load dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sentiment Analysis of Restaurant Reviews/Restaurant_Reviews.tsv', delimiter='\t')  # Replace with your dataset path

# Extract the feature (X) and target variable (y)
X = data['Review']
y = data['Liked']

# TF-IDF Vectorization with reduced features
tfidf_vectorizer = TfidfVectorizer(max_features=1300)
X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=0)

# Initialize Random Forest Classifier with regularization
random_forest_classifier = RandomForestClassifier(
    n_estimators=120,
    max_depth=23,
    min_samples_split=2,
    min_samples_leaf=10,
    random_state=0
)

# Train the Random Forest Classifier
random_forest_classifier.fit(X_train, y_train)

# Predict on training and testing data
y_train_pred = random_forest_classifier.predict(X_train)
y_test_pred = random_forest_classifier.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Random forest classifier:")
print("========================================================================================================================================================================================================================")
print("Training Accuracy:{}%".format(round(train_accuracy*100,2)))
print("========================================================================================================================================================================================================================")
print("Testing Accuracy:{}%".format(round(test_accuracy*100,2)))
print("========================================================================================================================================================================================================================")
diff = train_accuracy - test_accuracy
print("Difference between train and test accuracy:{}%".format(round(diff*100,2)))
print("========================================================================================================================================================================================================================")

import matplotlib.pyplot as plt
categories = ['Training Accuracy', 'Testing Accuracy', 'Accuracy difference']
values = [train_accuracy, test_accuracy, diff]
plt.bar(categories, values, color=['yellow', 'blue', 'red'])
plt.ylabel('Accuracy')
plt.title('Average Training Accuracy and Testing Accuracy')
plt.ylim(0, 1.0)
plt.show()