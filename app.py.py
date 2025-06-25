
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle

# Download stopwords if not already present
nltk.download('stopwords')

# Load the Dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')  # Adjust encoding as needed
except FileNotFoundError:
    print("Dataset file 'spam.csv' not found. Please make sure it exists in the working directory.")
    exit()

# Rename relevant columns
df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
df = df[['label', 'text']]

# Exploratory Data Analysis (optional visualization)
# sns.countplot(x='label', data=df)
# plt.show()

# Text Preprocessing Function
def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split(r'\W+', text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Clean the text
df['clean_msg'] = df['text'].apply(clean_text)

# Split data
X = df['clean_msg']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
# Trigger rebuild
