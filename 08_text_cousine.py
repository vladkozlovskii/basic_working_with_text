import numpy as np  
import pandas as pd
import re  
import scipy
import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
from pymystem3 import Mystem
from scipy.spatial import distance 

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
        text_df = pd.read_excel(path + 'text_for_analysis_rus.xlsx')
        text_df = pd.read_excel(path + 'text_for_analysis_eng.xlsx')
        stopwhile = 1
    except: 
        print('Error in entering the working folder name')

def check_language(text):
    rus = re.sub(r'[^а-я\']', ' ', text)
    rus = rus.split()
    rus = ''.join(rus)
    eng = re.sub(r'[^a-z\']', ' ', text)
    eng = eng.split()
    eng = ''.join(eng)

#    print(len(rus), rus)
#    print(len(eng), eng)

    if len(rus) == 0:
        print('Your input is english')
        lang = 1
    else: 
        if len(eng) == 0:
            print('Your input is russian')
            lang = 0
        else:
            if len(rus) > len(eng):
                print('Your input is generally russian text')
                lang = 0
            else: 
                print('Your input is generally english text')
                lang = 1
    return lang

def matrix_by_text(text, lang):
    # preprocessing like before making wordcloud
    if lang == 0:
        russian_stopwords = stopwords.words("russian")
        russian_stopwords.extend(['это', 'все', 'вс'])
        union = ' '.join(text)
        text_tokens = word_tokenize(union)
        text_tokens = [token.strip() for token in text_tokens if token not in russian_stopwords]
    else:
        english_stopwords = stopwords.words("english")
        union = ' '.join(text)
        text_tokens = word_tokenize(union)
        text_tokens = [token.strip() for token in text_tokens if token not in english_stopwords]
    
    # select unique tokens
    uniq_tokens = pd.Series(text_tokens).unique()

    # matrix for analysis in first dimension strings in text in second unique tokens
    matrix = np.zeros((len(text), len(uniq_tokens)))
    #print('Number of strings in text:', len(text))
    #print('Number of tokens in text:', len(text_tokens))
    #print('Number of unique tokens in text:', len(uniq_tokens))
    print('Matrix shape:', matrix.shape)
    
    # cycle for fill the matrix (if corresponding token is in sentence 1, if no 0)
    for sentence in range(len(text)):
        if lang == 0:
            single_sentence = re.split('[^а-я]', text[sentence])
        else:
            single_sentence = re.split('[^a-z]', text[sentence])   

        #print(single_sentence)
        for word in single_sentence:
            for i in range(len(uniq_tokens)):
                if word == uniq_tokens[i]:
                    #print(word, i)
                    matrix[sentence, i] = 1

    return matrix, uniq_tokens

# function for text filtration and lemmatization
def filtration(text, lang):
  text = text.lower()
  if lang == 0:
      text = ''.join(m.lemmatize(text))
      text = re.sub(r'[^а-я\']', ' ', text)
  else:
      text = ''.join(lemmatizer.lemmatize(text))
      text = re.sub(r'[^a-z\']', ' ', text)
  text = text.split()
  print('ok')
  return ' '.join(text)

# function for encoding of inputed text with unique tokens to compare it with matrix
def sentence_encoder(text, uniq_tokens, lang):
    text = filtration(text, lang)
    #print(text)
    if len(text) > len(uniq_tokens):
        text = text[:len(uniq_tokens)]
    encoded = np.zeros(len(uniq_tokens))
    for word in text.split():
        for i in range(len(uniq_tokens)):
            if word == uniq_tokens[i]:
                print(word, i)
                encoded[i] = 1
    return encoded, text

# function for counting cousine distance and define closest sentences from text and index or closest sentence
def nearest_phrase(encoded_input, matrix):

    index_list = []  # list for indexes
    distance_list = []  # list for cousine distances
    for i in range(matrix.shape[0]):  # cycle for compare encoded inputed phrase with phrases from file
         index_list.append(i)   
         distance_list.append(scipy.spatial.distance.cosine(encoded_input, matrix[i,:]))    # counting of cousine distance

    # table with results
    distance_table = pd.DataFrame([index_list, distance_list], index=['index', 'distance'])
    distance_table = distance_table.T
    #print(np.argmin(distance_table['distance']))
    needable_index = np.argmin(distance_table['distance'])
    distance_table = distance_table.sort_values(by='distance', ascending=True)
    #print(distance_table.head(5))
    
    return distance_table, needable_index

main_cycle = 0
while main_cycle != 1:
    print('Enter your phrase or press "Ctrl-C" to exit:')
    input_text = input()
    print('Inputed text:', input_text)
    lang = check_language(input_text)

    # open prepared file
    if lang == 0:
        text_df = pd.read_excel(path + 'text_for_analysis_rus.xlsx')
    else:
        text_df = pd.read_excel(path + 'text_for_analysis_eng.xlsx')    
    
    original = list(text_df['original'])
    text = list(text_df['processed'])

    matrix_a, uniq_tokens = matrix_by_text(text, lang)

    enc_input, phrase = sentence_encoder(input_text, uniq_tokens, lang)
    print('Processed inputed text:', phrase)

    distance_table, needable_index = nearest_phrase(enc_input, matrix_a)
    if distance_table['distance'][needable_index] > 0:
        print()
        print('5 closest phrase from text:')
        print()
        for i in distance_table['index'][:5]:
            print(text_df['original'][i], round(distance_table['distance'][i],2))
    else:
        print()
        print('There are no close phrases in text')