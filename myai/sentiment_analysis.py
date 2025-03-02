# import required libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# importing dataset
dataset=pd.read_csv('dataset.csv')
print(dataset.shape)
print(dataset.head())



# Segregating Dataset into Input and OutPUT

features=dataset.iloc[:,10].values
labels=dataset.iloc[:,1].values
print(features)
print(labels)

# Text preprocessing and Feature extraction

# TExt preprocessing
processed_features=[]

for sentence in range(0,len(features)):
  # Remove all the special charcters
  processed_feature=re.sub(r'\W',' ',str(features[sentence]))

  # remove all single characters
  processed_feature=re.sub(r'\s+[a-zA-Z]\s+',' ',processed_feature)

  # remove single characters from the start
  processed_feature=re.sub(r'^[a-zA-Z]\s+', ' ',processed_feature)

  # remove 'b' char from the text
  processed_feature=re.sub(r'^b\s+', '',processed_feature)

  # substitue multiple spaces with single space
  processed_feature=re.sub(r'\s+', ' ',processed_feature,flags=re.I)

  # Convert to lower case
  processed_feature=processed_feature.lower()
  processed_features.append(processed_feature)

# feature extraction from text

import nltk
nltk.download('stopwords')
vectorizer=TfidfVectorizer(max_features=2500,min_df=7,max_df=0.8,stop_words=stopwords.words('english'))
processed_features=vectorizer.fit_transform(processed_features).toarray()
print(processed_features)


# Splitting the dataset into train and test
X_train,X_test,Y_train,Y_test=train_test_split(processed_features,labels,test_size=0.2,random_state=0)

#Load Random forest alg
text_classifier_model=RandomForestClassifier(n_estimators=200,random_state=0)
text_classifier_model.fit(X_train,Y_train)


# predicting test data with trained model
predictions=text_classifier_model.predict(X_test)

# score of the model
print(accuracy_score(Y_test,predictions))



# sample new input for testing

new_input=["I hate NLP"]
new_input_vectorized=vectorizer.transform(new_input).toarray()

new_prediction=text_classifier_model.predict(new_input_vectorized)
print(new_prediction)


if(new_prediction=='negative'):
  print("unhappy")
else:
  print("happy")

