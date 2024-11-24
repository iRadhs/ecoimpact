from flask import Flask, request, jsonify
import json
import requests
import pandas as pd
from prophet import Prophet
from dataclasses import dataclass
from typing import List, Dict
import os

app = Flask(__name__)

@dataclass
class CarbonEmissionData:
    reportType: str
    subscriptionList: List[str]
    carbonScopeList: List[str]
    dateRange: Dict[str, str]

def get_bearer_token():
    url = 'https://carbonemissions-aba4e3gqdegrgjd4.canadacentral-01.azurewebsites.net/api/Admin/Connect'
    response = requests.post(url, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print("Failed to retrieve bearer token.")
        return None

bearer_token = get_bearer_token()

carbonemissions_url = 'https://carbonemissions-aba4e3gqdegrgjd4.canadacentral-01.azurewebsites.net/api/CarbonOptimization/CarbonEmissionData'

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
    
    response = requests.post(carbonemissions_url, headers={'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}, json=data)
    if response.status_code == 200:
        return emission_data, response.json()
    else:
        return None, None
response_data = get_carbon_emission_data()

@app.route('/forecast', methods=['POST'])
def forecast():
    emission_data, response_data = get_carbon_emission_data()
    
    if response_data:
        data = response_data['value']
        #response_data with 'date' and 'totalcarbonemission' values
        df = pd.DataFrame(data) 
        print("Dataframe:\n", df)

        # TotalCarbonEmission and Date columns for Prophet
        df['ds'] = pd.to_datetime(df['date'])  
        df['y'] = df['totalCarbonEmission']  
        df = df[['ds', 'y']]
        
        # Fit the Prophet model
        model = Prophet()
        model.fit(df)

        # forecasting (5 months ahead)
        future = model.make_future_dataframe(periods=5, freq='M')

        # Make predictions
        forecast = model.predict(future)
        print("\n")
        # Extracting forecasting for specific periods (1 month & 2 months)
        # Get the last date from the DataFrame
        last_date = df['ds'].max()

        # Forecast for the first month
        one_month_forecast = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= last_date + pd.DateOffset(months=1))].head(1)
        print("1 Month Forecast:\n", one_month_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        print("\n")

        # Forecast for the second month
        two_month_forecast = forecast[(forecast['ds'] > last_date + pd.DateOffset(months=1)) & (forecast['ds'] <= last_date + pd.DateOffset(months=2))].head(1)
        print("2 Months Forecast:\n", two_month_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        print("\n")

        # Forecast for the third month
        three_month_forecast = forecast[(forecast['ds'] > last_date + pd.DateOffset(months=2)) & (forecast['ds'] <= last_date + pd.DateOffset(months=3))].head(1)
        print("3 Months Forecast:\n", three_month_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Forecast for the fourth month
        four_month_forecast = forecast[(forecast['ds'] > last_date + pd.DateOffset(months=3)) & (forecast['ds'] <= last_date + pd.DateOffset(months=4))].head(1)
        print("4 Months Forecast:\n", four_month_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    return jsonify(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))