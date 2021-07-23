import os

import pandas as pd
import progress.bar
from tqdm import tqdm

# df=pd.read_excel("emails/Pisma_spam.xlsx",index_col=None)
# print( df)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from email_system.NecessaryEmail import NecessaryEmail
from spamfilter.Extractor import Extractor
from spamfilter.Tokenizer import Tokenizer
from spamfilter.classifiers.LSTMClassifier import LSTMClassifier
from spamfilter.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from spamfilter.classifiers.utils import int2label, label2int

navec_path = 'spamfilter/navec_hudlit_v1_12B_500K_300d_100q.tar'
dimensions = 300
sequence_length = 100
tokenizer = Tokenizer(navec_path, dimensions=dimensions, sequence_length=sequence_length)

path="lemmatized.csv"
if os.path.exists(path) and False:
    df=pd.read_csv(path)
else:
    extractor = Extractor(None)
    mails = extractor.from_mbox("emails/Myspam.mbox", True)
    spam = []
    for mail in mails:
        spam.append(mail)

    spam = extractor.get_dataframe(spam)
    spam_df = pd.read_excel("emails/Pisma_spam.xlsx", index_col=None)
    spam_df['label'] = 'spam'

    spam = pd.concat([spam, spam_df], join="inner", ignore_index=True)
    ham = pd.read_csv("emails/ham.CSV")
    # strip=NecessaryEmail.lstrip_subject
    # ham['subject'].apply(strip)
    ham['label'] = "ham"

    mails = pd.concat([spam, ham], join="inner", ignore_index=True)

    df = mails
    df.sample(frac=1)

    df['subject']=tokenizer.lemmatize_texts(df['subject'])
    df['text']=tokenizer.lemmatize_texts(df['text'])
# сохранить лемматизированный датасет
    df.to_csv(path)

tokenizer.fit_tokenizer(df['subject'])
tokenizer.fit_tokenizer(df['text'])
test_size = 0.2
x = tokenizer.texts_to_sequences(df['subject'])
y = [label2int[label] for label in df['label']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

naive_bayes_classifier = NaiveBayesClassifier()
naive_bayes_classifier.fit(X_train, y_train)

naive_bayes_predictions = []
# bar=progress.bar.IncrementalBar(max=len(X_test))
print("Naive Bayes")
for x in tqdm(X_test):
    # bar.next()
    naive_bayes_predictions.append(naive_bayes_classifier.predict(x))
# bar.finish()
print(accuracy_score(naive_bayes_predictions, y_test))

lstm_classifier = LSTMClassifier()
lstm_classifier.build_model(tokenizer.get_embedding_matrix(dimensions))
# lstm_classifier.save_model()
lstm_classifier.load_weights()
# loss, accuracy, precision, recall = lstm_classifier.train(X_train, y_train)
# print(f"Loss -- {loss}\nAccuracy -- {accuracy}\nPrecision -- {precision}\nRecall -- {recall}")

print("Lstm")

lstm_predictions=[]
for x in tqdm(X_test):
    # bar.next()
    lstm_predictions.append(label2int[lstm_classifier.predict(x)])
# bar.finish()
print(accuracy_score(lstm_predictions, y_test))
