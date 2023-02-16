
import numpy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
import streamlit as st


df1 = pd.read_csv('sinhala-hate-speech-dataset.csv')
df2 = pd.read_csv('Sinhala_Singlish_Hate_Speech.csv')

df2.columns= ["id","comment","label"]

df2['label'] = df2['label'].apply(lambda x: 1 if x == "YES" else 0)

df = pd.concat([df1, df2], sort=False)



df.isnull().sum()

import re

exclude = set(",.:;'\"-?!/´`%")
def remove_punctutation(text):
  return ''.join([(i if i not in exclude else " ") for i in text])

def remove_numbers(text):
  return ''.join(c for c in text if not c.isnumeric())

df['clean_data'] = df['comment'].apply(lambda x: remove_punctutation((x)))

df['cleand'] = df['clean_data'].apply(lambda x: remove_numbers(x))

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

df['tokens'] = df['cleand'].apply(word_tokenize)

with open("StopWords_425.txt", "r",encoding="utf-16") as file:
    # Read the contents of the file
    contents = file.read()
stop_word = contents.split()
stop_word = [word for word in stop_word if not any(char.isdigit() for char in word)]
print(stop_word)

df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop_word])

import nltk
from nltk.tokenize import word_tokenize

with open('Suffixes-413.txt', 'r', encoding='utf-16') as f:
    stemmed_words = f.readlines()

stemmed_words = [word for word in stemmed_words if not any(char.isdigit() for char in word)]
stemmed_words = [word.strip() for word in stemmed_words]
stemmed_words = set(stemmed_words)

def stem_word(word):
    if word in stemmed_words:
        return word
    else:
        return nltk.stem.PorterStemmer().stem(word)

df['cleaneddata'] = df['tokens'].apply(lambda x: [stem_word(word) for word in x])


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_word, token_pattern=r'\b\w+\b')),
    ('svm', SVC())
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.3)

pipeline.fit(X_train, y_train)



st.title("Sinhala Hate Speech Detector")

# Define the user input section
user_input = st.text_input("Enter a sentence")

# Define the model output section
if user_input:
    # Check if the sentence is hate or not
    user_pred = pipeline.predict([user_input])[0]
    if user_pred == 1:
        st.write("This sentence is hate.")
        add_to_df = st.selectbox("Is this correct?", ["Choose a Option","Yes", "No"],index=0)
        if add_to_df == "Yes":
            st.write("Thank you")
        else:
            processed_text = pd.Series(user_input)
            df = df.append({'comment': user_input, 'label': 0}, ignore_index=True)
            df.to_csv("sinhala-hate-speech-dataset", index=False)
            X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.3)
            X_train = X_train.append(processed_text, ignore_index=True)
            y_train = y_train.append(pd.Series([0]))
            pipeline.fit(X_train, y_train)
            st.write("Thank you for your contribution. We added that word into our system.")
    else:
        st.write("This sentence is not hate.")
        add_to_df = st.selectbox("Is this correct?", ["Choose a Option","Yes", "No"],index=0)
        if add_to_df == "Yes":
            st.write("Thank you")
        else:
            processed_text = pd.Series(user_input)
            df = df.append({'comment': user_input, 'label': 1}, ignore_index=True)
            df.to_csv("sinhala-hate-speech-dataset.csv",index=True)
            X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.3)
            X_train = X_train.append(processed_text, ignore_index=True)
            y_train = y_train.append(pd.Series([1]))
            pipeline.fit(X_train, y_train)
            st.write("Thank you for your contribution. We added that word into our system.")