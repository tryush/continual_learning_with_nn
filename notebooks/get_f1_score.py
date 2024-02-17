from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import numpy as np

def preprocess_data(data):
    max_words = 10000 
    max_length = 300  
 
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['text'])
 
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
 
    mlb = MultiLabelBinarizer()
    encoded_tags = mlb.fit_transform(data['tags'].apply(lambda x: set(x.split(', '))))
 
    return {'text': padded_sequences, 'tags': encoded_tags}

def get_ner(data, model):
    """ This Function takes a dataframe which has a text and tags column in it"""
    data['text'] = data['text'].astype(str)
    data['tags'] = data['tags'].apply(lambda x: re.sub(r"[^a-zA-Z_]", " ", x))
    data['tags'] = data['tags'].apply(lambda x: ', '.join(word.strip() for word in set(x.split())))
    
    test_data = preprocess_data(data)
    model = model
    tags_order = ['allergy_name', 'cancer', 'chronic_disease', 'treatment']
    predictions = model.predict(test_data['text'])
    binary_predictions = (predictions > 0.5).astype(int)
    true_labels = test_data['tags']
    
    f1_scores = []
    for i in range(len(tags_order)):
        tag_f1 = f1_score(true_labels[:, i], binary_predictions[:, i])
        f1_scores.append(tag_f1)
    
    print("\nOverall Classification Report:\n", classification_report(true_labels, binary_predictions, target_names=tags_order))
    
    f1_order = ['Allery F1', 'Cancer F1', 'Chronic Disease F1', 'Treatment F1']

    return dict(zip(f1_order, f1_scores))
 
    