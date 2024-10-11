import re  
import nltk
import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
from pymystem3 import Mystem
from wordcloud import WordCloud  
from datetime import datetime
from PIL import Image

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
m = Mystem()

stopwhile = 0
print()
print('Enter the folder name')

while stopwhile != 1:
    path = input()

    try:
        path = path.replace('\\', '/') + '/'
        print('Working folder:', path)

        stopwhile = 1
    except: 
        print('Error in entering the working folder name')

stopwhile = 0
print()
print('Enter the file name in the folder:', path)

while stopwhile != 1:
    filename = input()
    try:
        f = open(path + filename, 'r', encoding='utf8')
        text = f.read()  
        text
        stopwhile = 1
    except: 
        print('Error in entering the file name')

stopwhile = 0
print()
print('Enter the language of the text in the', path+filename)
print('0 - russian')
print('1 - english')

while stopwhile != 1:
    lang = int(input())
    try:
        if lang == 0:
            print('Language - russian')
            lang_str = 'rus'
            stopwhile = 1
        elif lang == 1:
            print('Language - english')
            lang_str = 'eng'
            stopwhile = 1
        else:
            print('wrong input')

    except: 
        print('Error in entering the language')

process = text
text_lower_by_sent = process.lower().split('\n')

def filtration(text, lang):
  if lang == 0:
      text = ''.join(m.lemmatize(text))
      text = re.sub(r'[^а-я\']', ' ', text)
  else:
      text = ''.join(lemmatizer.lemmatize(text))
      text = re.sub(r'[^a-z\']', ' ', text)
  text = text.split()
  print('ok')
  return ' '.join(text)

text_lower_by_sent_proc = [filtration(i, lang) for i in text_lower_by_sent]
union = ' '.join(text_lower_by_sent_proc)
text_tokens = word_tokenize(union)

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это', 'все', 'вс'])
english_stopwords = stopwords.words("english")

if lang == 0:
    text_tokens = [token.strip() for token in text_tokens if token not in russian_stopwords]
else:
    text_tokens = [token.strip() for token in text_tokens if token not in english_stopwords]

text_raw = " ".join(text_tokens)
time_now = str(datetime.now().day) + str(datetime.now().month) + str(datetime.now().year) + '_' + str(datetime.now().hour) + str(datetime.now().minute)

plt.figure(figsize=(10,10))
WC=WordCloud(width=700,height=500, max_words=300, min_font_size=5)
cloud=WC.generate(text_raw)
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(path + 'wc_' + lang_str + '_' + time_now + '.jpg')
plt.show
print('Image with wordcloud: wc_'+ lang_str + '_' + time_now + '.jpg')
print('saved in:', path)