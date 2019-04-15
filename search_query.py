import pandas as pd 
import sys
import numpy as np
from gensim import corpora, models, similarities, matutils
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
import itertools
import pandas as pd
import nltk, string
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
               
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
lsi = LsiModel.load('LsiModel')  # load model
print("===============================!!!!!!!!!!!!")
print(lsi)
lsi_index = MatrixSimilarity.load('MatrixSimilarity') # load index, (transformed corpus to LSI space and indexed it)
print(lsi_index)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens if item not in stop_words]

'''remove punctuation, lowercase, remove stopwords, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


def search_func(arxiv_id, num_results = 10):
	results = []
	query_details = df[df['id'] == str(arxiv_id)].reset_index(drop=True)
	try:
		query = query_details.loc[0, 'summary']
	except:
		print("Paper not present in the database.")
		return -1

	results.append([query_details.loc[0, 'title'], query_details.loc[0, 'link'], query_details.loc[0, 'summary']])

	vec_bow = dictionary.doc2bow(normalize(query))
	vec_lsi = lsi[vec_bow]  # convert the query to LSI space
	sims = lsi_index[vec_lsi]  # perform a similarity query against the corpus
	sims = sorted(enumerate(sims), key=lambda item: -item[1]) # sorted (document number, similarity score) 2-tuples

	# Print Query details
	# print("\nQuery:")
	# print("Title:", query_details.loc[0, 'title'])
	# print('Summary (query):', query)

	# Print top 15 results
	print("\n\nResults:\n")
	for i in range(num_results):
	    ind, score = sims[i]
	    # print("Title: ", df.loc[ind, 'title'])
	    # print("Link: ", df.loc[ind, 'link'])
	    # print("Summary :", df.loc[ind,'summary'])
	    # print(score)
	    # print("\n")
	    result = [df.loc[ind, 'title'], df.loc[ind, 'link'], df.loc[ind,'summary']]
	    results.append(result)
	return results

if __name__ == '__main__':
	query = sys.argv[1]
	num_results = int(sys.argv[2])
	# results is a list of list, results[0] has query details and results[1 to num_results] has final results.
	results = search_func(query, num_results)
	print(results)