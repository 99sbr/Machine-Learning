import tweepy
from tweepy import OAuthHandler
 
consumer_key = '***********'
consumer_secret = '**************'
access_token = '***********************'
access_secret = '*************************'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)
file = open("newfile.txt", "a")
for tweet in tweepy.Cursor(api.user_timeline).items():
	file.write(tweet.text)

f = open("newfile.txt")
data=f.read()
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
words=[i for i in data.lower().split() if i not in stop]
l=len(words)

import numpy as np
a=int(l/2)
words1=words[0:a]
words2=words[a+1:]
data1=' '.join(words1)
import re

from itertools import chain
text1 =re.sub(r"http\S+", "",data1).encode("utf-8")

import requests
payload1={"text":text1}
r = requests.post("http://text-processing.com/api/sentiment/", data=payload1)
x={}
x=r.json()
print(r.json)
data2=' '.join(words2)
import re

from itertools import chain
text2 =re.sub(r"http\S+", "",data2).encode("utf-8")

import requests
payload2={"text":text2}
r2 = requests.post("http://text-processing.com/api/sentiment/", data=payload2)
print(r2.content)
