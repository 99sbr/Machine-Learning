import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

user_id=test['User_ID']


train=train[['Description','Is_Response']]
test=test[['Description']]

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
train['Description'] =[re.sub('[^A-Za-z0-9- ]+', '',i) for i in train.Description if i not in stop]
test['Description'] =[re.sub('[^A-Za-z0-9- ]+', '', i) for i in test.Description if i not in stop]


max_features =100
tokenizer = Tokenizer(num_words=max_features, split=' ')

description=train['Description'].append(test['Description'])


tokenizer.fit_on_texts(description.values)
X = tokenizer.texts_to_sequences(train['Description'].values)
X = pad_sequences(X)

df_test = tokenizer.texts_to_sequences(test['Description'].values)
df_test = pad_sequences(df_test)


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(train['Is_Response']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)


batch_size = 32
model.fit(X_train, Y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)


score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

y_pred=model.predict(df_test)

submission=pd.DataFrame()
submission['User_Id']=user_id
submission['Is_Response']=y_pred
submission.to_csv('submission.csv',index=False)