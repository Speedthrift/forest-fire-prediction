from flask import Flask , redirect , render_template , request, url_for
from dotenv import load_dotenv
import os

import pickle, bz2
import pandas as pd
import numpy as np

import requests
from requests.structures import CaseInsensitiveDict

import warnings
warnings.filterwarnings("ignore")

#Environment variables for API keys
load_dotenv()


# Import Classification and Regression model file for Algeria
C_pickle = bz2.BZ2File(r'static\alg\Classification.pkl', 'rb')
alg_model_C = pickle.load(C_pickle)
R_pickle = bz2.BZ2File(r'static\alg\Regression.pkl', 'rb')
alg_model_R = pickle.load(R_pickle)


app = Flask(__name__)



@app.route('/')
def home():
	return render_template("indexMain.html")


@app.route('/learn-more')
def learn_more():
	return render_template("learn.html")

@app.route('/statistics')
def stats():
	return render_template("stats.html")


@app.route('/statistics/compare')
def compare():
	sections = [
	{
	"heading": "ResNet50 Aerial Image Analysis with Deep Learning (2024-2025)",
	"sentence": "Accuracy Score: 80.16% (after fine-tuning)",
	"subheading": "Limitations:",
	"points": [
	"Initial accuracy of 67.45 before fine-tuning indicates limited initial performance",
	"Requires high computational resources and memory for training large datasets",
	"Difficulty generalizing across different geographical regions and environmental conditions",
	"The model struggles with fixed spatial attributes rather than temporal patterns, necessitating additional architectural adjustments"
	]
	},
	{
	"heading": "LSTM and RNN for Smoke Detection using IoT Sensors (2024-2025)",
	"sentence": "Accuracy Score: 99.19% (validation accuracy)",
	"subheading": "Limitations",
	"points": [
	"Limited to temporal sensor data patterns and cannot capture spatial fire spread characteristics",
	"Requires well-preprocessed and normalized sensor data; sensitive to missing values",
	"May struggle with non-linear environmental relationships in complex real-world scenarios",
	"Computational complexity increases with larger temporal sequences, affecting real-time deployment"
	]
	},
	{
	"heading": "XGBoost and Random Forest for Wildfire Susceptibility Mapping (Turkey, 2022-2024)",
	"sentence": "Accuracy Scores: 88.8%",
	"subheading": "Limitations",
	"points": [
	"Reliance on accurate weather information means poor adaptability in areas with sparse or uneven meteorological data",
	"Cannot account for unpredictable human-caused fire ignitions",
	"Feature importance varies by region, requiring model retraining for new geographical areas",
	"Models struggle with rare fire events in sparsely vegetated regions"
	]
	},
	{
	"heading": "MA-Net Deep Learning for Large-Scale Fire Spread Prediction (2024)",
	"sentence": "Accuracy Score: F1-score of 0.64-0.68 depending on prediction day (1-5 days ahead)",
	"subheading": "Limitations",
	"points": [
	"Accuracy decreases significantly for longer prediction horizons (5 days) due to cumulative errors",
	"Dependent on high-resolution meteorological data; limited by weather forecast availability",
	"Poor performance when wind data is excluded (F1-score drops to 0.51)",
	"Trained models show poor generalization to different environmental regions (southern vs. northern zones)"
	]
	},
	{
	"heading": "Global Data-Driven Fire Activity Prediction using XGBoost (ECMWF, 2025)",
	"sentence": "Accuracy Scores: (Varies by region and data combination)",
	"subheading": "Limitations",
	"points": [
	"Quality of input data more important than model complexity; data-driven approaches fail without comprehensive datasets",
	"Significant data gaps: Real-time global fuel observations are scarce; fuel datasets rely on physical models",
	"Fire activity over-predicted in fuel-limited ecosystems when only weather data used",
	"Traditional fire weather indices inconsistently predict in desert/sparsely vegetated areas"
	]
	}
	]

	return render_template("comp.html", sections=sections)



@app.route('/ind/model-select')
def ind_home():
	return render_template("ind/ind_model_sel.html")


@app.route('/ind/predict-fire-lr' , methods = ['GET' , 'POST'])
def ind_predict_lr():
	msg = ""
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			folder_path = 'static/ind'
			file_name = 'finalModel_lr.sv'

			file_path = os.path.join(folder_path , file_name)

			predictor = pickle.load(open(file_path , 'rb'))
			prediction = predictor.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."

			return render_template("ind/predict_page_lr.html" , prediction_fire = msg, indback="Back")

		except:
			return render_template("ind/predict_page_lr.html" , prediction_fire = "Check the input again!!!", indback="Back")

	else:
		return render_template("ind/predict_page_lr.html" , prediction_fire = msg, indback="Back")


@app.route('/ind/predict-fire-dt' , methods = ['GET' , 'POST'])
def ind_predict_dt():
	msg = ""
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			folder_path = 'static/ind'
			file_name = 'finalModel_dt.sv'

			file_path = os.path.join(folder_path , file_name)

			predictor = pickle.load(open(file_path , 'rb'))
			prediction = predictor.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."
			return render_template("ind/predict_page_dt.html" , prediction_fire = msg, indback="Back")
			
		except:
			return render_template("ind/predict_page_dt.html" , prediction_fire = "Check the input again!!!", indback="Back")
	else:
		return render_template("ind/predict_page_dt.html" , prediction_fire = msg, indback="Back")


@app.route('/ind/predict-fire-rf' , methods = ['GET' , 'POST'])
def ind_predict_rf():
	msg = ""
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			folder_path = 'static/ind'
			file_name = 'finalModel_rf.sv'

			file_path = os.path.join(folder_path , file_name)

			predictor = pickle.load(open(file_path , 'rb'))
			prediction = predictor.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."
			return render_template("ind/predict_page_rf.html" , prediction_fire = msg, indback="Back")
			
		except:
			return render_template("ind/predict_page_rf.html" , prediction_fire = "Check the input again!!!", indback="Back")
	else:
		return render_template("ind/predict_page_rf.html" , prediction_fire = msg, indback="Back")


@app.route('/ind/predict-multi' , methods = ['GET' , 'POST'])
def ind_predict_multi():
	msg = ""
	if request.method == 'POST':
		try:
			vals=request.form['multi']
			model=request.form['mdname']
			points = vals.split("\r\n")
			res=[]
			# print(lst1)
			# temp = np.float64(request.form['Temperature']) / 100
			if model == '1':
				mod='Logistic Regression'
			elif model == '2':
				mod='Decision Tree'
			elif model == '3':
				mod='Random Forest'

			for i in range (0,len(points)):
				t,o,h = points[i].split(",")
				points[i]=f"Point {i+1} (Temp: {t}, Oxy: {o}, Hum: {h})"
				t = np.float64(t) / 100
				o = np.float64(o) / 100
				h = np.float64(h) / 100

				pred = ind_multi(t,o, h, model)

				if(pred > 0):
					msg = "Yes"
				else:
					msg = "No"

				res.append(msg)
			
			return render_template("ind/multires.html" , indback="Back", pts=points, res =res, md = mod)
			
		except:
			return render_template("ind/multi.html" , prediction_fire = "Check the input again!!!", indback="Back")
	else:
		return render_template("ind/multi.html" , prediction_fire = msg, indback="Back")



def ind_multi(tp,ox, hum, mod):
	folder_path = 'static/ind'
	if mod=="1":
		file_name = 'finalModel_lr.sv'
	elif mod=="2":
		file_name = 'finalModel_dt.sv'
	elif mod=="3":
		file_name = 'finalModel_rf.sv'

	file_path = os.path.join(folder_path , file_name)

	predictor = pickle.load(open(file_path , 'rb'))
	prediction = predictor.predict(np.array([[ox , tp , hum]]))[0]

	return prediction


@app.route('/ind/api/select-city' , methods = ['GET' , 'POST'])
def ind_api_sel():
	return render_template("ind/city_sel.html",indback="Back")


@app.route('/ind/api/result' , methods = ['GET' , 'POST'])
def ind_api_res():
	msg = ""
	if request.method == 'POST':
		try:
			city= request.form["city_name_text"]
			if not city:
				city= request.form["city_name"]
				if not city:
					# return render_template("ind/city_sel.html" , prediction_fire = "Select city!!!")
					# return redirect(url_for('city_page', prediction_fire="Select city!!!"))
					print("huh")
					return redirect(url_for('ind_api_sel'))


			model=request.form['mdname']
			if model == '1':
				mod='Logistic Regression'
			elif model == '2':
				mod='Decision Tree'
			elif model == '3':
				mod='Random Forest'
			try:
				lat, lon, e  = city_api(city)
				# print(e)
				# return render_template("ind/city_sel.html" , prediction_fire = e)			
				# return redirect(url_for('ind_api_sel'))

			except Exception as e:
				print(e)
				return redirect(url_for('ind_api_sel'))
				# return render_template("ind/city_sel.html" , prediction_fire = e)

			t,o,h = weather_api(lat, lon)

			# pred_res = ind_multi(t,o,h, model)

			pred = ind_multi(t,o, h, model)

			if(pred > 0):
				msg = "Yes"
			else:
				msg = "No"

			cont = f"""
					Selected city: {city}<br>
					Latitude: {lat}<br>
					Longitude: {lon}<br>
					Current Temperature: {t}℃<br>
					Current Humidity: {h}%<br>
					Predicted fire: {msg}<br>
					bruh; {pred}
					"""

			return render_template("ind/api_res.html", api_res=1, content=cont, model=mod,indback="Back")
		
		except Exception as e:
			# return render_template("ind/city_sel.html" , prediction_fire = "Check the input again!!!")
			print(e)
			return redirect(url_for('ind_api_sel'))



	else:
		# return render_template("ind/city_sel.html")
		return redirect(url_for('ind_api_sel'))


	# code= request.form["city_code"]
	

	# use ind multi here and resutns


def city_api(c):
		# url = "https://api.geoapify.com/v1/geocode/search?text=38%20Upper%20Montagu%20Street%2C%20Westminster%20W1H%201LJ%2C%20United%20Kingdom&apiKey={apik}"
		try:
			apik=os.environ.get("CITY_API")
			lst = c.split(" ")
			er=''
			if len(lst)>1:
				s = ""
				for i in range (0,len(lst)):
					s += lst[i]
					if i!=(len(lst)-1):
						s+= "%20"

				url = f"https://api.geoapify.com/v1/geocode/search?text={s}&apiKey={apik}"

			else:				
				url = f"https://api.geoapify.com/v1/geocode/search?text={c}&apiKey={apik}"

			headers = CaseInsensitiveDict()
			headers["Accept"] = "application/json"

			resp = requests.get(url, headers=headers)
			js = resp.json()
			longitude = js['features'][0]['properties']['lon']
			latitude = js['features'][0]['properties']['lat']

			return latitude, longitude, er
		
		except Exception as e:
			return 0,0, e
		# print(resp.status_code)


def weather_api(lat,lon):
	apik = os.environ.get("WEATHER_API")
	url=f"http://api.weatherapi.com/v1/current.json?key={apik}&q={lat},{lon}&aqi=no"
	resp = requests.get(url)
	g = resp.json() 
	temp = g['current']['temp_c']
	humid = g['current']['humidity'] 
	# oxy = 49.25
	oxy = 15

	return temp, oxy, humid


@app.route('/alg/model-select')
def alg_home():
	return render_template("alg/alg_model_sel.html")


@app.route('/alg/predict-fire-class', methods=['GET', 'POST'])
def alg_predict_class():
	if request.method == 'POST':
		try:
			#  reading the inputs given by the user
			Temperature=float(request.form['Temperature'])
			Wind_Speed =int(request.form['Ws'])
			FFMC=float(request.form['FFMC'])
			DMC=float(request.form['DMC'])
			ISI=float(request.form['ISI'])

			features = [Temperature, Wind_Speed,FFMC, DMC, ISI]

			Float_features = [float(x) for x in features]
			final_features = [np.array(Float_features)]
			prediction = alg_model_C.predict(final_features)[0]
			# prediction= float(pred)
			# log.info('Prediction done for Classification model')

			if prediction == 0:
				text = 'Forest is Safe!'
			else:
				text = 'Forest is in Danger!'
			return render_template('alg/alg_class.html', prediction_text1="{} --- Chance of Fire is {:.4f}".format(text, prediction), algback="Back")
		except Exception as e:
            # log.error('Input error, check input', e)
			# print('Input error, check input', e)
			# return render_template('alg/alg_class.html', prediction_text1=e)
			return render_template('alg/alg_class.html', prediction_text1="Check the Input again!!!", algback="Back")
	else:
		return render_template('alg/alg_class.html', algback="Back")


@app.route('/alg/predict-fire-reg', methods=['GET', 'POST'])
def alg_predict_reg():
	if request.method == 'POST':
		try:
			#  reading the inputs given by the user
			Temperature=float(request.form['Temperature'])
			Wind_Speed =int(request.form['Ws'])
			FFMC=float(request.form['FFMC'])
			DMC=float(request.form['DMC'])
			ISI=float(request.form['ISI'])

			features = [Temperature, Wind_Speed,FFMC, DMC, ISI]

			Float_features = [float(x) for x in features]
			final_features = [np.array(Float_features)]
			prediction = alg_model_R.predict(final_features)[0]


			# log.info('Prediction done for Regression model')

			if prediction > 15:
				return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(prediction), algback="Back")
			else:
				return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(prediction), algback="Back")

		except Exception as e:
			# log.error('Input error, check input', e)
			# print('Input error, check input', e)
			return render_template('alg/alg_reg.html', prediction_text2="Check the Input again!!!", algback="Back")
            
	else:
		return render_template('alg/alg_reg.html', algback="Back")
		


@app.route('/alg/predict-multi' , methods = ['GET' , 'POST'])
def alg_predict_multi():
	if request.method == 'POST':
		try:
			vals=request.form['multi']
			model=request.form['mdname']
			points = vals.split("\r\n")
			# res=[]
			tex=[]

			if model == '1':
				mod='Classification'
			elif model == '2':
				mod='Regression'

			for pt in range (0,len(points)):
				t,w,f,d,i = points[pt].split(",")
				points[pt]=f"Point {pt+1} (Temp: {t}, Wind Speed: {w}, FFMC: {f}, DMC: {d}, ISI: {i})"
				t = float(t) 
				w = int(w) 
				f = float(f) 
				d = float(d) 
				i = float(i) 
				pred = alg_multi(t,w,f,d,i,model)

				tex.append(pred)
				# res.append(chn)

			return render_template("alg/multires.html" , algback="Back", pts=points, tex=tex, md = mod)




		except Exception as e:
            # log.error('Input error, check input', e)
			# print('Input error, check input', e)
			# return render_template('alg/alg_class.html', prediction_text1=e)
			# return render_template('alg/multi.html', prediction_fire=e, algback="Back")
			return render_template('alg/multi.html', prediction_fire="Check the Input again!!!", algback="Back")
	else:
		return render_template('alg/multi.html', algback="Back")


def alg_multi(t,w,f,d,i, md):
	features = [t, w,f, d, i]

	Float_features = [float(x) for x in features]
	final_features = [np.array(Float_features)]


	# if md=='1':
	# 	chance = alg_model_C.predict(final_features)[0]
	# 	if chance == 0:
	# 		text = 'Safe!'
	# 	else:
	# 		text = 'Danger'
	# 	# return render_template('alg/alg_class.html', prediction_text1="{} --- Chance of Fire is {:.4f}".format(text, prediction))


	# elif md=='2':
	# 	chance = alg_model_R.predict(final_features)[0]
	# 	if chance > 15:
	# 		# return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(prediction))
	# 		text = 'Danger!'
	# 	else:
	# 		# return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(prediction))
	# 		text = 'Safe'

	if md=='1':
		chance = alg_model_C.predict(final_features)[0]
	elif md=='2':
		chance = alg_model_R.predict(final_features)[0]

	if chance <= 25:
		text = 'Safe'
	else:
		text = 'Danger!'
		
	# ch = "{:.2f}".format(chance)

	return text
	# prediction= float(pred)
	# log.info('Prediction done for Classification model')


if __name__ == '__main__':
	app.run(debug = True)




# @app.route('/create-model-ind' , methods = ['GET' , 'POST'])
# def ind_create_Model():
# 	score = ""
# 	if request.method == 'POST':
# 		algo = request.form.get('Model_Selection')
# 		file = pd.read_csv("static/ind/ForestFireDataSetUpdated2.1.csv")
# 		score = round((ind_modelScale(file , algo) * 100) , 2)
# 		return render_template('ind/model.html' , accuracy_model = 'Accuracy of the Selected Model - ' + str(score - 2) + '%')
	
# 	return render_template('ind/model.html' , accuracy_model = 'Select Model')


# def ind_modelScale(data:pd.DataFrame , model_name:str):
# 	features_list = ['Oxygen', 'Temperature', 'Humidity']
# 	for i in features_list:
# 		data[i] = MinMaxScaler().fit_transform(data[[i]])
# 	X = data.drop(columns= ["Area" , "Fire Occurrence"] , axis = 1)
# 	Y = data["Fire Occurrence"]
# 	X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state = 100 , test_size = 0.3)
# 	global modelClassifier
# 	if model_name == "DTC":
# 		modelClassifier = DecisionTreeClassifier(min_samples_leaf = 2 , ccp_alpha = 0.48 , random_state = 0)
# 	elif model_name == "RFC":
# 		modelClassifier = RandomForestClassifier(min_samples_leaf = 2 , ccp_alpha = 0.21 , random_state = 0)
# 	else:
# 		modelClassifier = LogisticRegression()
# 	modelClassifier.fit(X_train , Y_train)

# 	folder_path = 'static'
# 	file_name = 'finalModel.sv'

# 	file_path = os.path.join(folder_path , file_name)

# 	pickle.dump(modelClassifier , open(file_path , 'wb'))
# 	return accuracy_score(Y_test , modelClassifier.predict(X_test))

