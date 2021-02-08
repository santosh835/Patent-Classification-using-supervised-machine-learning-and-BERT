
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import string
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
stops = stopwords.words("english")
stemmer = PorterStemmer()

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}



#remove urls if any
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
#remove html tags 
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

#stemming
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

#stopwords removal
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def textPreProcessing(text):
    text=remove_urls(text)
    text=remove_html(text)
    text=remove_stopwords(text)
    text=text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#    text=stem_words(text)
    text=lemmatize_words(text)
    #remove numeric values
    text=text.str.replace('\d+', '')
    return text


def processDATA():
    df = pd.read_csv('data/PatentData.csv')
    
    df["abstracts"] = df["abstracts"].apply(lambda text: textPreProcessing(text))
    df['claims'] = df.claims.astype(str)
    df['description'] = df.description.astype(str)
    df["claims"] = df["claims"].apply(lambda text: textPreProcessing(text))
    df["description"] = df["description"].apply(lambda text: textPreProcessing(text))
    
    #lowercase the text
    df["abstracts"] = df["abstracts"].str.lower()
    df["claims"] = df["claims"].str.lower()
    df["description"] = df["description"].str.lower()
    df["patent_title"] = df["patent_title"].str.lower()
    
    
    label_dict={'A': 'HUMAN NECESSITIES', 'C': 'CHEMISTRY', 'G': 'PHYSICS', 'H': 'ELECTRICITY', 'F': 'MECHANICAL ENGINEERING', 'B': 'PERFORMING OPERATIONS; TRANSPORTING', 'D': 'TEXTILES', 'E': 'FIXED CONSTRUCTIONS'}
    df['classifications'] = df.classifications.replace(label_dict)
    
    df.to_csv("final_clean_data.csv")