from . import *  
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
	if not query:
		data = []
		output_message = 'HERE output_message'
	else:
		output_message = "Your search: " + query
		

		data = range(5)
	return render_template('search.html', 
		name=project_name, 
		netid_1=net_id_1, 
		netid_2=net_id_2, 
		netid_3=net_id_3, 
		netid_4=net_id_4, 
		netid_5=net_id_5, 
		output_message=output_message, 
		data=data)



