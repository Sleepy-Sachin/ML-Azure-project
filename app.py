import pandas as pd
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

df = pd.read_csv("flight_data2.csv")

categorical_features = ['origin', 'destination', 'weather', 'air_traffic_control', 'airline_operations', 'airport_operations', 'passenger_related', 'external_factors']
encoder = OneHotEncoder()
encoded_cols = encoder.fit_transform(df[categorical_features]).toarray()
df_encoded = pd.concat([df, pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_features))], axis=1)
df_encoded = df_encoded.drop(columns=categorical_features)

X = df_encoded.drop(columns=['delayed'])
y = df_encoded['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)  # Train the model

def predict_delay(input_data):
    input_df = pd.DataFrame([input_data], columns=['flight_number', 'scheduled_arrival', 'scheduled_departure', 'actual_departure', 'month', 'origin', 'destination', 'weather', 'air_traffic_control', 'airline_operations', 'airport_operations', 'passenger_related', 'external_factors'])

    encoded_input = encoder.transform(input_df[['origin', 'destination', 'weather', 'air_traffic_control', 'airline_operations', 'airport_operations', 'passenger_related', 'external_factors']]).toarray()
    input_df_encoded = pd.concat([input_df.drop(columns=['origin', 'destination', 'weather', 'air_traffic_control', 'airline_operations', 'airport_operations', 'passenger_related', 'external_factors']), pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['origin', 'destination', 'weather', 'air_traffic_control', 'airline_operations', 'airport_operations', 'passenger_related', 'external_factors']))], axis=1)

    prediction = model.predict(input_df_encoded)
    return 'Delayed' if prediction[0] == 1 else 'On Time'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    flight_number = request.form['flight_number']
    scheduled_arrival = int(request.form['scheduled_arrival'])
    scheduled_departure = int(request.form['scheduled_departure'])
    actual_departure = int(request.form['actual_departure'])
    month = int(request.form['month'])
    origin = request.form['origin']
    destination = request.form['destination']
    weather = request.form['weather']
    air_traffic_control = request.form['air_traffic_control']
    airline_operations = request.form['airline_operations']
    airport_operations = request.form['airport_operations']
    passenger_related = request.form['passenger_related']
    external_factors = request.form['external_factors']

    input_data = [flight_number, scheduled_arrival, scheduled_departure, actual_departure, month, origin, destination, weather, air_traffic_control, airline_operations, airport_operations, passenger_related, external_factors]
    prediction = predict_delay(input_data)
    
    return render_template('index.html', prediction_text=f"The flight is predicted to be: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
