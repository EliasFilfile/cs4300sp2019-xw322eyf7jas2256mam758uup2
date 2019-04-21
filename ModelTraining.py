import pandas as pd
from gensim import corpora, models, similarities, matutils
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity


# Read the pickle data
df = pd.read_pickle("ArxivData10k.pkl")

dictionary = corpora.Dictionary(df['summary_title_proc'])
corpus_gensim = [dictionary.doc2bow(doc) for doc in df['summary_title_proc']]

tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]

lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=250)
lsi.save('LsiModel')  # save model

lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
lsi_index.save('MatrixSimilarity') # save index

lsi = LsiModel.load('LsiModel')  # load model
lsi_index = MatrixSimilarity.load('MatrixSimilarity') # load index