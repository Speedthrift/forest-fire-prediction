from flask import Flask , redirect , render_template , request, url_for
from dotenv import load_dotenv
import os
import pickle, bz2
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

#Importing model files for India
file_path_lr = "static/ind/finalModel_lr.sv"
file_path_dt = "static/ind/finalModel_dt.sv"
file_path_rf = "static/ind/finalModel_rf.sv"

ind_model_lr = pickle.load(open(file_path_lr , 'rb'))
ind_model_dt = pickle.load(open(file_path_dt , 'rb'))
ind_model_rf = pickle.load(open(file_path_rf , 'rb'))

app = Flask(__name__)

#home page
@app.route('/')
def home():
	return render_template("indexMain.html")

#page about explaining terms
@app.route('/learn-more')
def learn_more():
	return render_template("learn.html")

#showing model accuracy and stuff
@app.route('/statistics')
def stats():
	return render_template("stats.html")

#comparison with previous works in domain
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

"""Indian forest fires"""
#model selection page for indian fires
@app.route('/ind/model-select')
def ind_home():
	return render_template("ind/ind_model_sel.html")

#for India prediction using Logistic Regression Model
@app.route('/ind/predict-fire-lr' , methods = ['GET' , 'POST'])
def ind_predict_lr():
	msg = ""
	#user has submitted form
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			prediction = ind_model_lr.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."
			return render_template("ind/predict_page_lr.html" , prediction_fire = msg, indback="Back")
		#in case of error in input validation
		except:
			return render_template("ind/predict_page_lr.html" , prediction_fire = "Check the input again!!!", indback="Back")

	#user has only loaded the page, not submitted the form
	else:
		return render_template("ind/predict_page_lr.html" , prediction_fire = msg, indback="Back")

#for India prediction using Decision Tree Model
@app.route('/ind/predict-fire-dt' , methods = ['GET' , 'POST'])
def ind_predict_dt():
	msg = ""
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			prediction = ind_model_dt.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."
			return render_template("ind/predict_page_dt.html" , prediction_fire = msg, indback="Back")
			
		except:
			return render_template("ind/predict_page_dt.html" , prediction_fire = "Check the input again!!!", indback="Back")
	else:
		return render_template("ind/predict_page_dt.html" , prediction_fire = msg, indback="Back")

#for India prediction using Random Forest Model
@app.route('/ind/predict-fire-rf' , methods = ['GET' , 'POST'])
def ind_predict_rf():
	msg = ""
	if request.method == 'POST':
		try:
			temp = np.float64(request.form['Temperature']) / 100
			oxygen = np.float64(request.form['Oxygen']) / 100
			humid = np.float64(request.form['Humidity']) / 100

			prediction = ind_model_rf.predict(np.array([[oxygen , temp , humid]]))[0]

			if(prediction > 0):
				msg = "Fire can occur under these conditions."
			else:
				msg = "No Worry! Fire will not take place."
			return render_template("ind/predict_page_rf.html" , prediction_fire = msg, indback="Back")
			
		except:
			return render_template("ind/predict_page_rf.html" , prediction_fire = "Check the input again!!!", indback="Back")
	else:
		return render_template("ind/predict_page_rf.html" , prediction_fire = msg, indback="Back")

#for India prediction for multiple points using user selected Model
@app.route('/ind/predict-multi' , methods = ['GET' , 'POST'])
def ind_predict_multi():
	msg = ""
	if request.method == 'POST':
		try:
			vals=request.form['multi']
			model=request.form['mdname']
			points = vals.split("\r\n")
			res=[]
			#mapping numbers to model name for later using in form
			if model == '1':
				mod='Logistic Regression'
			elif model == '2':
				mod='Decision Tree'
			elif model == '3':
				mod='Random Forest'

			#iterating over the points in the input and assigning prediction scores to each of them
			for i in range (0,len(points)):
				t,o,h = points[i].split(",")
				points[i]=f"Point {i+1} (Temp: {t}, Oxy: {o}, Hum: {h})"	#changes list element for better readability in frontend
				t = np.float64(t) / 100
				o = np.float64(o) / 100
				h = np.float64(h) / 100
				#predicting for each point
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

#used when multiple points
def ind_multi(tp,ox, hum, mod):
	if mod=="1":
		prediction = ind_model_lr.predict(np.array([[ox , tp , hum]]))[0]
	elif mod=="2":
		prediction = ind_model_dt.predict(np.array([[ox , tp , hum]]))[0]
	elif mod=="3":
		prediction = ind_model_rf.predict(np.array([[ox , tp , hum]]))[0]

	return prediction

#page for selecting city, based on which current prediction will be given
@app.route('/ind/api/select-city' , methods = ['GET' , 'POST'])
def ind_api_sel():
	return render_template("ind/city_sel.html",indback="Back")

#route for displaying prediction based on current data
@app.route('/ind/api/result' , methods = ['GET' , 'POST'])
def ind_api_res():
	msg = ""
	if request.method == 'POST':
		try:
			#in case of names entered by user in both text box and dropdown selected, entered tet takes priority
			city= request.form["city_name_text"]
			if not city:
				city= request.form["city_name"]
				if not city:	#means user did not enter any city or select one, so return them to the page
					return redirect(url_for('ind_api_sel'))

			model=request.form['mdname']
			if model == '1':
				mod='Logistic Regression'
			elif model == '2':
				mod='Decision Tree'
			elif model == '3':
				mod='Random Forest'

			try:
				lat, lon  = city_api(city)
			except Exception as e:	#incase of any error in fetching data from api
				# print(e)
				return redirect(url_for('ind_api_sel'))

			t,o,h = weather_api(lat, lon)
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
					"""
			return render_template("ind/api_res.html", api_res=1, content=cont, model=mod,indback="Back")
		
		except Exception as e:
			return redirect(url_for('ind_api_sel'))
	else:
		return redirect(url_for('ind_api_sel'))

#for fetching coordinates (latitude & longitude) based on city name entered
def city_api(c):
		try:
			apik=os.environ.get("CITY_API")
			lst = c.split(" ")
			#optimising entered city name to work with url, in case of citties with space in their name
			#if splitted list has more than one element => city has space in name
			#so, for searching with API, 'New Delhi' has to be made into 'New%20Delhi'
			if len(lst)>1:
				s = ""
				for i in range (0,len(lst)):
					s += lst[i]
					if i!=(len(lst)-1):
						s+= "%20"
				url = f"https://api.geoapify.com/v1/geocode/search?text={s}&apiKey={apik}"

			else:				
				url = f"https://api.geoapify.com/v1/geocode/search?text={c}&apiKey={apik}"

			#procedure from API documentation
			headers = CaseInsensitiveDict()
			headers["Accept"] = "application/json"
			resp = requests.get(url, headers=headers)
			js = resp.json()

			longitude = js['features'][0]['properties']['lon']
			latitude = js['features'][0]['properties']['lat']
			return latitude, longitude
		
		except Exception:
			return 0,0
		# print(resp.status_code)

#for fetching weather conditions based on latitude and longitude
def weather_api(lat,lon):
	apik = os.environ.get("WEATHER_API")
	url=f"http://api.weatherapi.com/v1/current.json?key={apik}&q={lat},{lon}&aqi=no"
	resp = requests.get(url)
	g = resp.json() 
	temp = g['current']['temp_c']
	humid = g['current']['humidity'] 
	#no reliable source for Oxygen concentration found
	oxy = 15
	return temp, oxy, humid


"""Algerian Forest fires"""

#model selection page for Algeria
@app.route('/alg/model-select')
def alg_home():
	return render_template("alg/alg_model_sel.html")

#for predicting Alregia fires with Classification Model
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

			if prediction == 0:
				text = 'Forest is Safe!'
			else:
				text = 'Forest is in Danger!'
			return render_template('alg/alg_class.html', prediction_text1="{} --- Chance of Fire is {:.4f}".format(text, prediction), algback="Back")
			
		except Exception as e:
			return render_template('alg/alg_class.html', prediction_text1="Check the Input again!!!", algback="Back")
	else:
		return render_template('alg/alg_class.html', algback="Back")

#for predicting Algeria fires with Regression Model
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

			if prediction > 15:
				return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(prediction), algback="Back")
			else:
				return render_template('alg/alg_reg.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(prediction), algback="Back")

		except Exception as e:
			return render_template('alg/alg_reg.html', prediction_text2="Check the Input again!!!", algback="Back")
            
	else:
		return render_template('alg/alg_reg.html', algback="Back")
		
#for predicting Algeria fires for multiple points using mdoelselected by user
@app.route('/alg/predict-multi' , methods = ['GET' , 'POST'])
def alg_predict_multi():
	if request.method == 'POST':
		try:
			vals=request.form['multi']
			model=request.form['mdname']
			points = vals.split("\r\n")
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
			return render_template("alg/multires.html" , algback="Back", pts=points, tex=tex, md = mod)

		except Exception as e:
			return render_template('alg/multi.html', prediction_fire="Check the Input again!!!", algback="Back")
	else:
		return render_template('alg/multi.html', algback="Back")

#used for predicting for multiple points in Algeria
def alg_multi(t,w,f,d,i, md):
	features = [t, w,f, d, i]

	Float_features = [float(x) for x in features]
	final_features = [np.array(Float_features)]

	if md=='1':
		chance = alg_model_C.predict(final_features)[0]
	elif md=='2':
		chance = alg_model_R.predict(final_features)[0]

	if chance <= 25:
		text = 'Safe'
	else:
		text = 'Danger!'
	return text

if __name__ == '__main__':
	app.run(debug = True)