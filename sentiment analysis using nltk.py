#Cleaning text steps
# 1) Create a text file and take text from it
# 2) Convert the letter into lowercase ( 'Apple' is not equal to 'apple'
# 3) Remove punctuations like  .,!? etc. ( Hi! This is buildwithpython. ) 
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import sys # to access the system
import cv2

positive = cv2.imread("happy_emoji.jpg", cv2.IMREAD_ANYCOLOR)
negative =  cv2.imread("sad_emoji.jpeg", cv2.IMREAD_ANYCOLOR)
neutral =  cv2.imread("neutral_emoji.png", cv2.IMREAD_ANYCOLOR)

text = open('read.txt',encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))
 
 
tokenized_words =  word_tokenize(cleaned_text,"english")



final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)

w = Counter(emotion_list)

print(w)

def sentiment_analyse(sentiment_text):
    score  = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        print("Negative Sentiment")
        cv2.imshow("Negative", negative) 
    elif pos>neg:
        print("Positive Sentiment")
        cv2.imshow("Positive", positive) 
    else:
        print("Neutral Vibe")
        cv2.imshow("Neutral", neutral)        

sentiment_analyse(cleaned_text)


fig,ax1 = plt.subplots()
ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()