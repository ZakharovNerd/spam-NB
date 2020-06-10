import pandas as pd
import string
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix




df=pd.read_csv('SMSSpamCollection.csv',sep='\t')
df.columns = ['count','message']
df['message'] = df.message.str.lower()
df['message'] = df.message.str.replace('[{0}]*'.format(string.punctuation), '')
df['message'] = df['message'].str.replace('\d+', '')



x = df[:3898]
y = df[3898:-1]




        

def make_Dictionary(train_dir):
    all_words = []
    for mail in train_dir:
        words = mail.split()
        all_words += words
        dictionary = Counter(all_words)
    return dictionary



#разделение даты 

def extract_features(mail_dir, dictionary):
#mail_dir - начальный необработанный файл. От него мы берем количество строк
#dictionary - обработанный словарь с количеством вхождения слов в mail_dir
    features_matrix = np.zeros((mail_dir.shape[0], len(dictionary.keys())))
    mail_dir_id = 0
    for m in mail_dir:
        words = m.split()
        for word in words:
            for i in range(len(dictionary)):
                features_matrix[mail_dir_id,i] = words.count(word)
        mail_dir_id += 1
    return features_matrix

#разделяем дату
x_ham = x[x['count']=='ham']
x_spam = x[x['count']=='spam']
x = pd.concat([x_ham, x_spam])

y_ham, y_spam = y[y['count']=='ham'], y[y['count']=='spam']



dicy = make_Dictionary(x['message'])


# можелька
dictionary = make_Dictionary(df)
train_labels = np.zeros(x.shape[0])
train_labels[x_ham.shape[0]:x.shape[0] - 1] = 1
train_matrix = extract_features(x['message'], dicy)

model1 = MultinomialNB()

model1.fit(train_matrix,train_labels)



test_matrix = extract_features(y['message'], dicy)
test_labels = np.zeros(y.shape[0])
test_labels[y_ham.shape[0]:y.shape[0] - 1] = 1
result1 = model1.predict(test_matrix)

print(confusion_matrix(test_labels,result1))




















