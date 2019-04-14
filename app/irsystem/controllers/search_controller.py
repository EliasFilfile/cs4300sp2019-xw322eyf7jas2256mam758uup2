from . import * 
import json 
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
		result1 = {}
		result1 ["title"] = "An Optimal Control View of Adversarial Machine Learning "
		result1["abstract"]= "I describe an optimal control view of adversarial machine learning, where the\\ndynamical system is the machine learner, the input are adversarial actions, and\\nthe control costs are defined by the adversary's goals to do harm and be hard\\nto detect. This view encompasses many types of adversarial machine learning,\\nincluding test-item attacks, training-data poisoning, and adversarial reward\\nshaping. The view encourages adversarial machine learning researcher to utilize\\nadvances in control theory and reinforcement learning."
		result1["link"] = ""
		result2 = {}
		result2 ["title"] = "An Optimal Control View of Adversarial Machine Learning "
		result2["abstract"]= "I describe an optimal control view of adversarial machine learning, where the\\ndynamical system is the machine learner, the input are adversarial actions, and\\nthe control costs are defined by the adversary's goals to do harm and be hard\\nto detect. This view encompasses many types of adversarial machine learning,\\nincluding test-item attacks, training-data poisoning, and adversarial reward\\nshaping. The view encourages adversarial machine learning researcher to utilize\\nadvances in control theory and reinforcement learning."
		


		data = [result1, result2]
	return render_template('search.html', 
		name=project_name, 
		netid_1=net_id_1, 
		netid_2=net_id_2, 
		netid_3=net_id_3, 
		netid_4=net_id_4, 
		netid_5=net_id_5, 
		output_message=output_message, 
		data=data)



