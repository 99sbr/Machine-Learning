import json
import requests
TOKEN="EAACEdEose0cBAMiD4s13kZCvfNs0iKGZBREoOZChE0rtJZCQVHpL0R4bInvJmktqOczzWhktG2GTbXFYr3qvKewr1aS2Ajmy5eCZC9Tf6E4GZAj6bswMGTrgnyrwF3ZAvKAZB8BxyOPVe2GdsniCM0uNgFGt7q9oOnrTEGWXVJgA9gZDZD"

parameters = {'access_token': TOKEN}
r = requests.get('https://graph.facebook.com/me/feed', params=parameters)
result = json.loads(r.text)
words=[]
for i in range(0,len(result['data'])):
    try:
        
        l=len(result['data'][i]['comments']['data'])
        for j in range(0,l):
            words.append(result['data'][i]['comments']['data'][j]['message'])
    except:
        try:
            words.append(result['data'][i]['description'])
        except:
            try:
                words.append(result['data'][i]['message'])
            except:
                words.append(result['data'][16]['privacy']['description'])
 
data=[]
data=' '.join(words)
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
text=[i for i in data.lower().split() if i not in stop]
l=len(text)
import numpy as np
a=int(l/2)
text1=text[0:a]
text2=text[a+1:]
corpse1=' '.join(text1)
import re
clean_text1 =re.sub(r"http\S+", "",corpse1).encode("utf-8")
payload1={"text":clean_text1}
r = requests.post("http://text-processing.com/api/sentiment/", data=payload1)
d={}
x=r.json()
d=x['probability']
corpse2=' '.join(text2)
clean_text2 =re.sub(r"http\S+", "",corpse2).encode("utf-8")
payload2={"text":clean_text2}
r2 = requests.post("http://text-processing.com/api/sentiment/", data=payload2)
e={}
y=r2.json()
e=y['probability']
positive_probability=d.get('pos')*e.get('pos')
negative_probability=d.get('neg')*e.get('neg')
neutral_probability=d.get('neutral')*e.get('neutral')

positive_score=positive_probability*35
negative_score=negative_probability*35
neutral_score=neutral_probability*30
print(positive_score)
print(negative_score)
print(neutral_score)