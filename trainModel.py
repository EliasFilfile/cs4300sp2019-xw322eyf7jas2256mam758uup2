import numpy as np
import pandas as pd
import nltk, string
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from gensim import corpora, models, similarities, matutils
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity

download('punkt')
download('stopwords')
               
stop_words = stopwords.words('english')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation)

# Read the processed 'title' and 'summary' data
df = pd.read_pickle("ArxivData10k.pkl")
# Creating the dictionary
dictionary = corpora.Dictionary(df['summary_proc'])
# Load model and index

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens if item not in stop_words]

'''remove punctuation, lowercase, remove stopwords, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

df = pd.read_pickle("ArxivData10k.pkl")
dictionary = corpora.Dictionary(df['summary_proc'])
corpus_gensim = [dictionary.doc2bow(doc) for doc in df['summary_proc']]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=250)
lsi.save('LsiModel')  # save model
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
lsi_index.save('MatrixSimilarity') # save index

ind = 16
query = df.loc[ind,'summary']
vec_bow = dictionary.doc2bow(normalize(query))
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
sims = lsi_index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims))[:10])  # print (document_number, document_similarity) 2-tuples