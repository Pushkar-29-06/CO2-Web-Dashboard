from flask import Blueprint, render_template, jsonify, send_file, request
import pandas as pd
import os
import io
import base64
import joblib
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Set style for plots
plt.style.use('ggplot')  # Using 'ggplot' style which is similar to seaborn
sns.set_theme(style="whitegrid")

main = Blueprint('main', __name__)

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'co2_emission_predictor.joblib')

# Load data
def load_data():
    try:
        # Get the absolute path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'processed_emissions.csv')
        data_path = os.path.normpath(data_path)
        
        print(f"Attempting to load data from: {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Print debug info
        print(f"Successfully loaded data. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head(2).to_string()}")
        
        return df
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise  # Re-raise the exception to be handled by the caller

@main.route('/api/cities')
def get_cities():
    """API endpoint to get list of all available cities"""
    df = load_data()
    cities = sorted(df['City'].unique().tolist())
    return jsonify(cities)

@main.route('/api/compare-cities')
def compare_cities():
    """API endpoint to compare two cities"""
    try:
        print("\n--- Compare Cities Request ---")
        city1 = request.args.get('city1')
        city2 = request.args.get('city2')
        
        print(f"Received request to compare: {city1} and {city2}")
        
        if not city1 or not city2:
            error_msg = 'Both city1 and city2 parameters are required'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        try:
            df = load_data()
            print(f"Loaded data with {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Available cities: {df['City'].unique().tolist()}")
            
            # Add City_Lower column if it doesn't exist
            if 'City_Lower' not in df.columns:
                print("Adding City_Lower column...")
                df['City_Lower'] = df['City'].str.strip().str.lower()
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': 'Failed to load data', 'details': str(e)}), 500
        
        # Get list of available cities for case-insensitive matching
        available_cities = df['City'].unique()
        city_map = {str(city).strip().lower(): str(city).strip() for city in available_cities}
        
        print(f"City map (lowercase to actual): {city_map}")
        
        # Find matching cities (case-insensitive)
        city1_lower = str(city1).strip().lower()
        city2_lower = str(city2).strip().lower()
        
        if city1_lower not in city_map or city2_lower not in city_map:
            missing = []
            if city1_lower not in city_map:
                missing.append(city1)
            if city2_lower not in city_map:
                missing.append(city2)
            error_msg = f"City not found: {', '.join(missing)}"
            print(error_msg)
            return jsonify({
                'error': error_msg,
                'missing_cities': missing,
                'available_cities': sorted([city_map[k] for k in city_map])
            }), 404
            
        # Get actual city names with correct case
        actual_city1 = city_map[city1_lower]
        actual_city2 = city_map[city2_lower]
        
        print(f"Found cities: {actual_city1} and {actual_city2}")
        
        # Debug: Print city data
        print(f"Data for {actual_city1}:")
        print(df[df['City_Lower'] == city1_lower].to_string())
        print(f"\nData for {actual_city2}:")
        print(df[df['City_Lower'] == city2_lower].to_string())
        
        # Get historical trends for both cities
        city1_trend = df[df['City_Lower'] == city1_lower].sort_values('Year')
        city2_trend = df[df['City_Lower'] == city2_lower].sort_values('Year')
        
        print(f"\nCity 1 trend data shape: {city1_trend.shape}")
        print(f"City 2 trend data shape: {city2_trend.shape}")
        
        # Check if we have data for the cities
        if city1_trend.empty or city2_trend.empty:
            error_msg = f"No data found for one or both cities. {actual_city1}: {not city1_trend.empty}, {actual_city2}: {not city2_trend.empty}"
            print(error_msg)
            return jsonify({'error': error_msg}), 404
        
        # Helper function to convert numpy types to native Python types
        def convert_to_python(val):
            if hasattr(val, 'item'):  # For numpy types
                return val.item()
            if isinstance(val, (int, float)):
                return float(val)
            return val
            
        # Get the latest data for each city
        def get_latest_data(city_trend):
            latest = city_trend.sort_values('Year', ascending=False).iloc[0]
            return {
                'CO2_Emission_kt': convert_to_python(round(float(latest.get('CO2_Emission_kt', 0)), 2)),
                'AQI_Index': convert_to_python(round(float(latest.get('AQI_Index', 0)), 1)),
                'Industrial_Activity_Score': convert_to_python(round(float(latest.get('Industrial_Activity_Score', 0)), 1))
            }
            
        try:
            city1_data = get_latest_data(city1_trend)
            city2_data = get_latest_data(city2_trend)
            
            # Prepare trends data
            def prepare_trends(trend_df):
                years = trend_df['Year'].astype(str).str.strip().tolist()
                emissions = [convert_to_python(round(float(x), 2)) for x in trend_df['CO2_Emission_kt'].tolist()]
                aqi = [convert_to_python(round(float(x), 1)) for x in trend_df['AQI_Index'].tolist()]
                return {
                    'years': years,
                    'emissions': emissions,
                    'aqi': aqi
                }
                
            city1_trends = prepare_trends(city1_trend)
            city2_trends = prepare_trends(city2_trend)
            
            # Prepare response
            response = {
                'emissions': {
                    actual_city1: city1_data['CO2_Emission_kt'],
                    actual_city2: city2_data['CO2_Emission_kt']
                },
                'aqi': {
                    actual_city1: city1_data['AQI_Index'],
                    actual_city2: city2_data['AQI_Index']
                },
                'industry': {
                    actual_city1: city1_data['Industrial_Activity_Score'],
                    actual_city2: city2_data['Industrial_Activity_Score']
                },
                'trends': {
                    actual_city1: dict(zip(
                        city1_trends['years'],
                        city1_trends['emissions']
                    )),
                    actual_city2: dict(zip(
                        city2_trends['years'],
                        city2_trends['emissions']
                    ))
                },
                'aqi_trends': {
                    actual_city1: dict(zip(
                        city1_trends['years'],
                        city1_trends['aqi']
                    )),
                    actual_city2: dict(zip(
                        city2_trends['years'],
                        city2_trends['aqi']
                    ))
                }
            }
            
            print(f"Prepared response with keys: {list(response.keys())}")
            print(f"Sample trend data for {actual_city1}:", list(response['trends'][actual_city1].items())[:3])
            print(f"Sample AQI data for {actual_city2}:", list(response['aqi_trends'][actual_city2].items())[:3])
            
        except Exception as e:
            error_msg = f"Error preparing response data: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'error': 'Failed to prepare comparison data',
                'details': str(e)
            }), 500
        
        print(f"Prepared response with keys: {list(response.keys())}")
        print(f"Trends keys: {list(response['trends'].keys())}")
        print(f"Sample trend data: {list(response['trends'][actual_city1].items())[:3]}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error in compare_cities: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@main.route('/compare')
def compare():
    city1 = request.args.get('city1')
    city2 = request.args.get('city2')
    
    if not city1 or not city2:
        return redirect('/')
        
    return render_template('comparison.html')

@main.route('/')
def index():
    df = load_data()
    
    # Generate plot URLs
    plots = {
        'emissions_trend': create_emissions_trend_plot(df),
        'city_emissions': create_city_emissions_plot(df),
        'aqi_emissions': create_aqi_emissions_plot(df),
        'industry_emissions': create_industry_emissions_plot(df)
    }
    
    # Get summary statistics
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    total_emissions = latest_data['CO2_Emission_kt'].sum()
    avg_aqi = latest_data['AQI_Index'].mean()
    max_emission_city = latest_data.loc[latest_data['CO2_Emission_kt'].idxmax()]
    
    # Create correlation heatmap
    plots['correlation_heatmap'] = create_correlation_heatmap(df)
    
    return render_template('index.html', 
                         plots=plots,
                         total_emissions=total_emissions,
                         avg_aqi=round(avg_aqi, 1),
                         top_city={
                             'name': max_emission_city['City'],
                             'emission': max_emission_city['CO2_Emission_kt'],
                             'aqi': max_emission_city['AQI_Index']
                         },
                         cities_count=df['City'].nunique())
                         
def create_correlation_heatmap(df):
    """Create a correlation heatmap for numerical features with enhanced styling."""
    try:
        # Select only numerical columns for correlation
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numerical_cols].corr()
        
        # Set up the matplotlib figure
        plt.figure(figsize=(14, 12))
        
        # Create a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create heatmap with better styling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(
            correlation_matrix, 
            mask=mask,  # Only show lower triangle
            cmap=cmap, 
            vmin=-1, vmax=1, center=0,
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot=True, 
            fmt=".2f",
            annot_kws={"size": 9}
        )
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # Add title and adjust layout
        plt.title('Correlation Heatmap of City Metrics', 
                 pad=20, 
                 fontsize=16, 
                 fontweight='bold',
                 color='#2c3e50')
        
        # Add a descriptive caption
        plt.figtext(0.5, 0.01, 
                   'Correlation values range from -1 (perfect negative) to 1 (perfect positive).', 
                   ha='center', 
                   fontsize=10,
                   color='#666666')
        
        # Improve layout to prevent label cutoff
        plt.tight_layout()
        
        # Add colorbar with a label
        cbar = ax.collections[0].colorbar
        cbar.set_label('Correlation Strength', rotation=270, labelpad=20, fontsize=10)
        
        # Save the plot to a bytes buffer with higher DPI for better quality
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Encode the image to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        return None

def create_emissions_trend_plot(df):
    """Create emissions trend plot and return as base64 encoded image."""
    plt.figure(figsize=(10, 6))
    
    # Group by year and calculate mean emissions
    yearly_avg = df.groupby('Year')['CO2_Emission_kt'].mean()
    
    # Create the plot
    ax = sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, 
                     marker='o', linewidth=2.5, color='#10B981')
    
    # Customize the plot
    plt.title('Average CO2 Emissions Trend (2017-2024)', fontsize=14, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('CO2 Emissions (kt)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_city_emissions_plot(df):
    """Create city emissions plot and return as base64 encoded image."""
    plt.figure(figsize=(10, 6))
    
    # Get latest year data and sort
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].sort_values('CO2_Emission_kt', ascending=False).head(10)
    
    # Create the plot
    ax = sns.barplot(x='CO2_Emission_kt', y='City', data=latest_data, hue='City', palette='viridis', legend=False)
    
    # Customize the plot
    plt.title(f'Top 10 Cities by CO2 Emissions ({latest_year})', fontsize=14, pad=20)
    plt.xlabel('CO2 Emissions (kt)', fontsize=12)
    plt.ylabel('')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_aqi_emissions_plot(df):
    """Create AQI vs Emissions scatter plot and return as base64 encoded image."""
    plt.figure(figsize=(10, 6))
    
    # Get latest year data
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    # Create the plot
    scatter = plt.scatter(
        x=latest_data['CO2_Emission_kt'],
        y=latest_data['AQI_Index'],
        s=latest_data['Population_M'] * 100,  # Scale point size by population
        c=latest_data['Industrial_Activity_Score'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Industrial Activity Score', rotation=270, labelpad=15)
    
    # Customize the plot
    plt.title(f'AQI vs CO2 Emissions by City ({latest_year})', fontsize=14, pad=20)
    plt.xlabel('CO2 Emissions (kt)', fontsize=12)
    plt.ylabel('AQI Index', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_industry_emissions_plot(df):
    """Create industrial activity vs emissions plot and return as base64 encoded image."""
    try:
        # Create a new figure with a specific size
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with regression line
        sns.regplot(
            x='Industrial_Activity_Score',
            y='CO2_Emission_kt',
            data=df,
            scatter_kws={'alpha':0.5, 'color':'#4CAF50'},
            line_kws={'color':'#2E7D32'}
        )
        
        # Customize the plot
        plt.title('Industrial Activity vs CO₂ Emissions', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Industrial Activity Score', fontsize=12, labelpad=10)
        plt.ylabel('CO₂ Emissions (kt)', fontsize=12, labelpad=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Close the plot to free memory
        plt.close()
        
        # Encode the image to base64
        return base64.b64encode(img.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating industry emissions plot: {str(e)}")
        return None

@main.route('/predict')
def predict_co2():
    """Render the CO2 prediction page with city selection."""
    df = load_data()
    cities = sorted(df['City'].unique().tolist())
    latest_year = df['Year'].max()
    
    return render_template('predict.html', 
                         cities=cities,
                         latest_year=latest_year)

@main.route('/api/city-prediction/<city_name>')
def get_city_prediction(city_name):
    """Generate and return a prediction plot for the selected city."""
    try:
        df = load_data()
        
        # Get city data
        city_data = df[df['City'] == city_name]
        if city_data.empty:
            return jsonify({'status': 'error', 'message': 'City not found'}), 404
            
        # Get latest year data for this city
        latest_year = city_data['Year'].max()
        latest_data = city_data[city_data['Year'] == latest_year].iloc[0]
        
        # Create a simple prediction visualization
        plt.figure(figsize=(10, 6))
        
        # Plot historical data
        years = city_data['Year'].unique()
        emissions = city_data.groupby('Year')['CO2_Emission_kt'].mean()
        
        # Simple linear regression for prediction
        from sklearn.linear_model import LinearRegression
        X = years.reshape(-1, 1)
        y = emissions.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 5 years
        future_years = np.arange(years.min(), years.max() + 6).reshape(-1, 1)
        predicted = model.predict(future_years)
        
        # Plot historical data
        plt.plot(years, y, 'o-', label='Historical', color='#4CAF50', linewidth=2)
        
        # Plot prediction
        future_years_flat = future_years.flatten()
        plt.plot(future_years_flat, predicted, '--', 
                label='Prediction', color='#FF5722', linewidth=2)
        
        # Highlight the selected city's latest data point
        plt.scatter([latest_year], [latest_data['CO2_Emission_kt']], 
                   color='red', s=100, zorder=5, 
                   label=f'Latest ({latest_year})')
        
        # Add prediction details
        prediction_2025 = model.predict([[2025]])[0]
        plt.axhline(y=prediction_2025, color='#9E9E9E', linestyle=':', alpha=0.7)
        plt.text(2025.2, prediction_2025, 
                f'2025: {prediction_2025:,.0f} kt',
                va='center', ha='left', color='#FF5722')
        
        # Style the plot
        plt.title(f'CO₂ Emissions Trend & Prediction\n{city_name}', fontsize=14, pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('CO₂ Emissions (kt)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # Prepare response
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Get city stats with native Python types
        stats = {
            'current_emission': float(latest_data['CO2_Emission_kt']),
            'predicted_2025': float(round(prediction_2025, 2)),
            'aqi': float(latest_data.get('AQI_Index', 0)) if pd.notna(latest_data.get('AQI_Index')) else 'N/A',
            'industry_score': float(latest_data.get('Industrial_Activity_Score', 0)) if pd.notna(latest_data.get('Industrial_Activity_Score')) else 'N/A',
            'vehicle_density': float(latest_data.get('Vehicle_Density_per_km2', 0)) if pd.notna(latest_data.get('Vehicle_Density_per_km2')) else 'N/A',
            'population': float(latest_data.get('Population_M', 0)) if pd.notna(latest_data.get('Population_M')) else 'N/A',
            'forest_cover': float(latest_data.get('Forest_Cover_pct', 0)) if pd.notna(latest_data.get('Forest_Cover_pct')) else 'N/A'
        }
        
        return jsonify({
            'status': 'success',
            'image': image_base64,
            'stats': stats,
            'city': city_name
        })
        
    except Exception as e:
        print(f"Error generating prediction for {city_name}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
    return render_template('predict.html',
                         plots={
                             'city_emissions': create_city_emissions_plot(df)
                         },
                         total_emissions=total_emissions,
                         avg_aqi=round(avg_aqi, 1),
                         top_city={
                             'name': max_emission_city['City'],
                             'emission': max_emission_city['CO2_Emission_kt'],
                             'aqi': max_emission_city['AQI_Index']
                         },
                         latest_year=latest_year,
                         cities_count=df['City'].nunique())

@main.route('/api/predict-co2', methods=['POST'])
def api_predict_co2():
    """API endpoint for CO2 prediction with historical data visualization."""
    try:
        print("Received prediction request")
        # Check if request has JSON data
        if not request.is_json:
            error_msg = 'Missing JSON in request: Request must be JSON'
            print(f"Error: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'details': 'Request must be JSON'
            }), 400
            
        # Get input data from request
        data = request.get_json()
        print(f"Request data: {data}")
        
        # Check for required fields
        required_fields = [
            'Year', 'Population_M', 'Vehicle_Density_per_km2',
            'Industrial_Activity_Score', 'Forest_Cover_pct'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'Missing required fields: {missing_fields}'
            print(f"Error: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'missing_fields': missing_fields
            }), 400
        
        try:
            # Load the model and data
            model = joblib.load(MODEL_PATH)
            df = load_data()
            
            # Get historical data for the selected city (if available)
            city_data = df[df['City'] == data.get('City', '')]
            historical_data = []
            
            # Prepare input features in the correct order expected by the model
            input_features = [
                float(data['Year']),
                float(data['Population_M']),
                float(data['Vehicle_Density_per_km2']),
                float(data['Industrial_Activity_Score']),
                float(data['Forest_Cover_pct']),
                float(data.get('SO2_Annual_Avg_ugm3', 35.0)),
                float(data.get('NO2_Annual_Avg_ugm3', 40.0)),
                float(data.get('PM10_Annual_Avg_ugm3', 90.0)),
                float(data.get('PM2.5_Annual_Avg_ugm3', 60.0)),
                float(data.get('AQI_Index', 65))
            ]
            
            # Make prediction
            prediction = model.predict([input_features])[0]
            
            # Generate historical data for the chart
            if not city_data.empty:
                # Get historical data for the city
                historical_data = city_data[['Year', 'CO2_Emission_kt']].sort_values('Year').to_dict('records')
            else:
                # Generate sample historical data if city not found
                current_year = int(data['Year'])
                base_emission = float(prediction) * 0.8  # Start at 80% of predicted value
                for year in range(current_year - 5, current_year):
                    historical_data.append({
                        'Year': year,
                        'CO2_Emission_kt': base_emission * (0.9 + 0.1 * (year - (current_year - 5)) / 5)
                    })
            
            # Add current prediction to historical data
            historical_data.append({
                'Year': int(data['Year']),
                'CO2_Emission_kt': float(prediction)
            })
            
            # Generate chart image
            plt.figure(figsize=(10, 5))
            years = [str(item['Year']) for item in historical_data]
            emissions = [item['CO2_Emission_kt'] for item in historical_data]
            
            plt.plot(years, emissions, 'b-o', linewidth=2, markersize=8, label='Historical Data')
            
            # Highlight the prediction point
            plt.scatter([str(data['Year'])], [prediction], color='red', s=100, zorder=5, 
                       label='Prediction')
            
            plt.title('CO2 Emissions Trend', fontsize=14, fontweight='bold')
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('CO2 Emissions (kt)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Encode the image to base64
            chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Return prediction with chart data
            return jsonify({
                'status': 'success',
                'prediction': round(float(prediction), 2),
                'chart_image': chart_image,
                'historical_data': historical_data
            })
            
        except Exception as model_error:
            import traceback
            return jsonify({
                'status': 'error',
                'message': 'Error during prediction',
                'details': str(model_error),
                'traceback': traceback.format_exc()
            }), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 400
