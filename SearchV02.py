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
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

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


'''this version will just combine all the abstracts into one and use that as the query_details'''
def get_vector1(queries):
    summary = ""
    for q in queries:
        summary += q["summary"]

    vec_bow = dictionary.doc2bow(normalize(summary))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    return vec_lsi

'''this version creates the vector based on the equally weighted query vectors'''
def get_vector2(queries):
    vec_lsi = None
    length = len(queries)
    for q in queries:
        if vec_lsi == None:
            vec_bow = dictionary.doc2bow(normalize(q['summary']))
            vec_lsi = scale(lsi[vec_bow], length)
        else:
            vec_bow = dictionary.doc2bow(normalize(q['summary']))
            vec_lsi = scale(lsi[vec_bow], length, vec_lsi)
    return vec_lsi

def scale(vector, length, vec_lsi = None):
    for pos in range(len(vector)):
        vector[pos] = (pos, (vector[pos][1] / length))
    if vec_lsi != None:
        for pos in range(len(vec_lsi)):
            vec_lsi[pos] = (pos, (vector[pos][1] + vec_lsi[pos][1]))
    return vector


'''Given the list of arxiv id's returns the query vector for the combined query'''
def get_query_vector(arxiv_ids):
    query_details = []
    for x in arxiv_ids:
        query_details.append(df[df['id'] == str(x)].reset_index(drop=True))
    queries = []
    try:
        for query in query_details: #fix starting here
            q = {}
            q['title'] = query.loc[0, 'title']
            q['link'] = query.loc[0, 'link']
            q['summary'] = query.loc[0, 'summary']
            queries.apppend(q)
    except:
        queries = []
        for pos,query in enumerate(query_details):
            q = {}

            url = "https://arxiv.org/abs/" + arxiv_ids[pos].replace("arXiv:","")
            data = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(data, 'html.parser')
            q['title'] = soup.find("meta",  property="og:title")['content']
            q['link'] = soup.find("meta",  property="og:url")['content']
            q['summary'] = soup.find("meta",  property="og:description")['content']
            queries.append(q)

    vec_lsi = get_vector2(queries)
    return vec_lsi


'''Given a query vector and the number of desired results, will return the top
num_results relevant documents'''
def get_results(query, num_results = 10):
    results = []
    #return [type(query)]
    sims = lsi_index[query]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1]) # sorted (document number, similarity score) 2-tuples

    # Print top 10 results
    print("\n\nResults:\n")
    for i in range(num_results):
        ind, score = sims[i]
        citation = find_citation(df.loc[ind, 'title'])
        print("Title: ", df.loc[ind, 'title'])
        print("Link: ", df.loc[ind, 'link'])
        print("Summary :", df.loc[ind,'summary'])
        print(score)
        print("\n")
        result = [df.loc[ind, 'title'], df.loc[ind, 'link'], df.loc[ind,'summary'], citation, score]
        results.append(result)
    return results


''' Takes in a list of the queried arxiv id's and returns the top 10 most relevant
documents from the databases. Default uses indirect relevance feedback, but can
have none with feedback = False, or direct with relevant = {set of relevent docs}
and/or irrelevant = {set of irrelevant docs}
'''
def multi_search(arxiv_ids, num_results = 10, feedback = False, relevant = [], irrelevant = [], alpha = .7, beta = .3, gamma = .3):
    try:
        vec_lsi = get_query_vector(arxiv_ids)
        results = get_results(vec_lsi, num_results)
        #return [results]
        #with feedback
        if feedback:
            if relevant != [] and irrelevant != []:
                return "wtffff1"
                d1 = get_query_vector(relevant)
                d2 = get_query_vector(irrelevant)
                q1 = addVector(addVector(multVector(vec_lsi, alpha), multVector(vec_lsi, beta)),  multVector(d2, gamma), sub=True)
            elif relevant != []:
                return "wtffff12"
                d1 = get_query_vector(relevant)
                q1 = addVector(multVector(vec_lsi, alpha), multVector(d1, beta))
            elif irrelevant != []:
                return "wtffff123"
                d2 = get_query_vector(irrelevant)
                q1 = addVector(multVector(vec_lsi, alpha), multVector(d2, gamma), sub=True)
            else:
                ids = []
                for r in results[:3]:
                    ids.append(r[1][21:])
                d = get_query_vector(ids)
                #return addVector
                a = multVector(vec_lsi, alpha)
                b = multVector(d, beta)
                q1 = addVector(a, b)
            results = get_results(q1, num_results)
        return results
    except:
        return -1

def multVector(vec, scalar):
    for pos in range(len(vec)):
        vec[pos] = (pos, (vec[pos][1] * scalar))
    return vec

def addVector(vec1, vec2, sub=False):
    #return vec[0]
    if sub:
        for pos in range(len(vec1)):
            vec1[pos] = (pos, (vec1[pos][1] - vec2[pos[1]]))
    else:
        for pos in range(len(vec1)):
            vec1[pos] = (pos, (vec1[pos][1] + vec2[pos][1]))
    return vec1

def find_citation(title):
    search_term = title.lower()

    headers = {
        # Request headers
        'Ocp-Apim-Subscription-Key': 'bbb41ec4267c4e1d9f98e9056b1b989e',
    }
    params = urllib.parse.urlencode({
        # Request parameters
        'expr': "Ti="+ "\'" + search_term + "\'",
        'model': 'latest',
        'count': '10',
        'attributes' : 'CC, Ti'
    })
    try:
        conn = http.client.HTTPSConnection('api.labs.cognitive.microsoft.com')
        conn.request("GET", "/academic/v1.0/evaluate?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read()
        jdata = json.loads(data.decode('utf-8'))
        base = jdata["expr"]
        cited_by = jdata["entities"][0]['CC']
        ti = base[4:len(base)-1]
        print(cited_by)
        print(ti)
        conn.close()
        return cited_by
    except Exception:
        return 0


def id_retriever(title):
    new_title = title.replace(" ", "+").replace('\n', '')
    url = "http://export.arxiv.org/api/query?search_query=ti:" + "\"" + new_title + "\"&start=0&max_results=1"
    data = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(data, 'html.parser')
    #print(soup.prettify())
    #return [url]
    id = soup.find_all('id')[1].contents[0][21:]
    ti = soup.find_all('title')[1].contents[0].replace('\n', '')
    # print(id)
    # print(ti)
    assert(ti.replace(' ', '') == title.replace(' ', ''))
    return id

def multi_title_search(titles, num_results = 10, feedback = False, relevant = [], irrelevant = [], alpha = .7, beta = .3, gamma = .3):
    try:
        ids = []
        for t in titles:
            ids.append(id_retriever(t))
        return multi_search(ids, num_results, feedback, relevant, irrelevant, alpha, beta, gamma)
    except:
        return -1


def search_keywords(keywords, num_results=5):
    try:
        #keywords = keywords.join("+") #this one if keywords is a list
        keywords = keywords.replace(' ', '+') #this one if keywords is a string of keywords seperated by spaces
        url = "http://export.arxiv.org/api/query?search_query=all:" + keywords + "&start=0&max_results=" + str(num_results)
        data = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(data, 'html.parser')
        ids = soup.find_all('id')[1:]
        ti = soup.find_all('title')[1:]
        abst = soup.find_all('summary')
        results = []
        for i in range(len(ids)):
            title = ti[i].contents[0].replace('\n', '')
            citation = find_citation(title)
            result = [title, ids[i].contents[0], abst[i].contents[0] , citation]
            results.append(result)
        return results
    except:
        return -1


def search(arxiv_id, num_results=5):
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


    # Preprocess the query
    vec_bow = dictionary.doc2bow(normalize(summary))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = lsi_index[vec_lsi]  # perform a similarity query against the corpus
    # sorted (document number, simlenilarity score) 2-tuples
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
        return [ind]

        if tot_results == num_results:
            break
        if df.loc[ind, 'title'] == title:
            continue

        citation = find_citation(df.loc[ind, 'title'])

        print("Title: ", df.loc[ind, 'title'])
        print("Link: ", df.loc[ind, 'link'])
        print("Summary :", df.loc[ind, 'summary'])
        print(score)
        print("\n")
        result = [df.loc[ind, 'title'],
                  df.loc[ind, 'link'], df.loc[ind, 'summary'], citation, score]
        results.append(result)
        tot_results += 1
    return results


if __name__ == '__main__':
    query = sys.argv[1]
    num_results = int(sys.argv[2])
    # results is a list of list, results[0] has query details and results[1 to num_results] has final results.
    results = search(query, num_results)
    # print(results)
