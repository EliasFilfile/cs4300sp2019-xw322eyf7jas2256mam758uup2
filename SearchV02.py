import pandas as pd
import sys
import numpy as np
from gensim import corpora, models, similarities, matutils
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
import itertools
import pandas as pd
import nltk
import string
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen, Request

# download('punkt')
# download('stopwords')

stop_words = stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation)

# Read the processed 'title' and 'summary' data
df = pd.read_pickle("ArxivData10k.pkl")
# Creating the dictionary
dictionary = corpora.Dictionary(df['summary_proc'])
# Load model and index
lsi = LsiModel.load('LsiModel')  # load model
# load index, (transformed corpus to LSI space and indexed it)
lsi_index = MatrixSimilarity.load('MatrixSimilarity')


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens if item not in stop_words]


'''remove punctuation, lowercase, remove stopwords, stem'''


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


def citation_search(title):

    title = title.replace(" ", "+")
    url_base = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C33&q='
    url = url_base + "\"" + title + "\""

    #print(url)

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    # query the website and return the html to the variable ‘page’
    page = str(urlopen(req).read())
    start = page.find("Cited by ")
    end = page[start:].find("<")

    # no citation found
    if (start == -1) or (end == -1):
        return -1

    # citation found
    else:
        return int(page[start+9:start+end])


def search(arxiv_id, num_results=10):
    results = []
    query_details = df[df['id'] == str(arxiv_id)].reset_index(drop=True)

    try:
        title = query_details.loc[0, 'title']
        link = query_details.loc[0, 'link']
        summary = query_details.loc[0, 'summary']
    except:
        url = "https://arxiv.org/abs/" + arxiv_id.replace("arXiv:", "")
        data = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(data, 'html.parser')
        title = soup.find("meta",  property="og:title")['content']
        link = soup.find("meta",  property="og:url")['content']
        summary = soup.find("meta",  property="og:description")['content']

    # Append the query
    # results.append([title, link, summary])

    # Preprocess the query
    vec_bow = dictionary.doc2bow(normalize(summary))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = lsi_index[vec_lsi]  # perform a similarity query against the corpus
    # sorted (document number, similarity score) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # Print Query details
    print("\nQuery:")
    print("Title:", title)
    print('Summary (query):', summary)

    # Print top 15 results
    print("\n\nResults:\n")
    tot_results = 0
    for i in range(num_results+1):

        ind, score = sims[i]
        if tot_results == num_results:
            break
        if len(query_details) == 1 and df.loc[ind, 'id'] == query_details.loc[0, 'id']:
            continue

        citation = citation_search(df.loc[ind, 'title'])

        print("Title: ", df.loc[ind, 'title'])
        print("Link: ", df.loc[ind, 'link'])
        print("Summary :", df.loc[ind, 'summary'])
        print(score)
        print("\n")
        result = [df.loc[ind, 'title'],
                  df.loc[ind, 'link'], df.loc[ind, 'summary'], citation]
        results.append(result)
        tot_results += 1
    return results


if _name_ == '_main_':
    query = sys.argv[1]
    num_results = int(sys.argv[2])
    # results is a list of list, results[0] has query details and results[1 to num_results] has final results.
    results = search(query, num_results)
    # print(results)
