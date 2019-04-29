from . import *
import json
import SearchV02 as sq
import re
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder


project_name = "Reseach Paper Recommendation System"
net_id_1 = "Xueyuan Wang: xw322"
net_id_2 = "Maxim Markovics: mam758"
net_id_3 = "Elias Filfile: eyf7"
net_id_4 = "Unmesh Padalkar: uup2"
net_id_5 = "Jayadeep Shitole: jas2256"

@irsystem.route('/', methods=['GET'])
def search():

	query = request.args.get('search')
	qtype = request.args.get('optradio')

	if not query:
		data = []
		output_message = 'HERE output_message'
	else:
		print(query)
		formatted = query.replace(' ', '').split(';')
		error_message = 'ERROR'
		output_message = "Your search: " + query
		#formatted = re.split('; +', query)
		#formatted = query.split(' ')
		#formatted = query

		'''Uncomment out the code below and fill in wiht logic when available'''
		if (qtype == 'arXivID'):
			formatted = query.replace(' ', '').split(';')
			error_message = 'Sorry, we couldn\'t find that ID, check to make sure it is correct'
			results = sq.multi_search(formatted, num_results=5)
		elif (qtype == 'title'):
			formatted = re.split('; +', query)
			error_message = 'Sorry, we couldn\'t find that Title, check to make sure it is correct'
			results = sq.multi_title_search(formatted, num_results=5)
		elif (qtype == 'keywords'):
			formatted = query
			error_message = 'Sorry, we couldn\'t find anything matching those keywords, check to make sure they is correct'
			results = sq.search_keywords(formatted, num_results=5)

		if results == -1:
			data = [error_message]
		else:
			data = results
	return render_template('search.html',
		name=project_name,
		netid_1=net_id_1,
		netid_2=net_id_2,
		netid_3=net_id_3,
		netid_4=net_id_4,
		netid_5=net_id_5,
		output_message=output_message,
		data=data)
