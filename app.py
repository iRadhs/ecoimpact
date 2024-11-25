import logging
from flask import Flask, jsonify, request
import requests
import pandas as pd
from prophet import Prophet
from dataclasses import dataclass
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# CarbonEmissionData dataclass
@dataclass
class CarbonEmissionData:
    reportType: str
    subscriptionList: List[str]
    carbonScopeList: List[str]
    dateRange: Dict[str, str]

# API URLs
TOKEN_URL = 'https://carbonemissions-aba4e3gqdegrgjd4.canadacentral-01.azurewebsites.net/api/Admin/Connect'
CARBON_EMISSIONS_URL = 'https://carbonemissions-aba4e3gqdegrgjd4.canadacentral-01.azurewebsites.net/api/CarbonOptimization/CarbonEmissionData'

def get_bearer_token():
    try:
        response = requests.post(TOKEN_URL, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an error for bad responses
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving bearer token: {e}")
        return None

bearer_token = get_bearer_token()

def get_carbon_emission_data():
    data = {
        "reportType": "MonthlySummaryReport",
        "subscriptionList": [
            "3df2bc8c-3c22-42b0-824a-10289255cf2a"
        ],
        "carbonScopeList": [
            "Scope1",
            "Scope2",
            "Scope3"
        ],
        "dateRange": {
            "start": "2023-11-01",
            "end": "2024-11-01"
        }
    }
    
    # Create an instance of CarbonEmissionData
    emission_data = CarbonEmissionData(
        reportType=data["reportType"],
        subscriptionList=data["subscriptionList"],
        carbonScopeList=data["carbonScopeList"],
        dateRange=data["dateRange"]
    )
    
    try:
        response = requests.post(
            CARBON_EMISSIONS_URL,
            headers={'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'},
            json=data
        )
        response.raise_for_status()
        return emission_data, response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving carbon emission data: {e}")
        return None, None

@app.route('/forecast', methods=['POST'])
def forecast():
    emission_data, response_data = get_carbon_emission_data()
    
    if not response_data:
        return jsonify({"error": "Failed to retrieve carbon emission data"}), 500
    
    data = response_data.get('value')
    
    if not data:
        return jsonify({"error": "No data found in the response"}), 404

    # Create DataFrame from the response data
    df = pd.DataFrame(data)
    
    if df.empty:
        return jsonify({"error": "DataFrame is empty"}), 404

    logging.info("Dataframe created successfully:\n%s", df)

    # Prepare DataFrame for Prophet
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['totalCarbonEmission']
    df = df[['ds', 'y']]

    # Fit the Prophet model
    model = Prophet()
    
    try:
        model.fit(df)
        
        # Forecasting (5 months ahead)
        future = model.make_future_dataframe(periods=5, freq='M')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Extract forecasts for specific periods (1 to 4 months ahead)
        last_date = df['ds'].max()
        
        forecasts = {}
        
        for i in range(1, 5):
            month_forecast = forecast[(forecast['ds'] > last_date + pd.DateOffset(months=i-1)) & 
                                      (forecast['ds'] <= last_date + pd.DateOffset(months=i))]
            forecasts[f"{i}_month_forecast"] = month_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        
        # Save the entire forecast to a CSV file
        forecast.to_csv('forecast_results.csv', index=False)

        return jsonify(forecasts)

    except Exception as e:
        logging.error(f"Error during forecasting: {e}")
        return jsonify({"error": "Forecasting failed"}), 500

if __name__ == '__main__':
   app.run(debug=True)