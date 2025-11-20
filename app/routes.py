from flask import Blueprint, render_template, jsonify, send_file, request, url_for, redirect
import pandas as pd
import os
import io
import base64
import time
import uuid
from datetime import datetime
import joblib
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from collections import defaultdict
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')  # Using 'ggplot' style which is similar to seaborn
sns.set_theme(style="whitegrid")

main = Blueprint('main', __name__, static_folder='static')

def save_plot(fig, filename_prefix):
    """Save plot to static/images and return the URL."""
    try:
        # Create static/images directory if it doesn't exist
        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.makedirs(static_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.png"
        filepath = os.path.join(static_dir, filename)
        
        # Save the plot
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        # Return the URL for the saved image
        return url_for('static', filename=f'images/{filename}')
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        return None

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'co2_emission_predictor.joblib')

def load_data():
    """
    Load and validate the emissions data from Excel file.
    Returns a clean pandas DataFrame.
    """
    try:
        # Get the absolute path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'maharashtra_city_emissions_2017_2024.xlsx')
        data_path = os.path.normpath(data_path)
        
        print(f"\n=== Loading data from: {data_path} ===")
        
        # Check if file exists and is readable
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        if not os.access(data_path, os.R_OK):
            raise PermissionError(f"No read permissions for file: {data_path}")
        
        # Read the Excel file
        df = pd.read_excel(data_path)
        
        # Clean column names (remove extra spaces and newlines)
        df.columns = df.columns.str.strip()
        
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("The Excel file is empty")
        
        # Define expected columns and their data types
        expected_columns = {
            'Year': 'int64',
            'State': 'object',
            'City': 'object',
            'CO2_Emission_kt': 'float64',
            'SO2_Annual_Avg_ugm3': 'float64',
            'NO2_Annual_Avg_ugm3': 'float64',
            'PM10_Annual_Avg_ugm3': 'float64',
            'PM2.5_Annual_Avg_ugm3': 'float64',
            'AQI_Index': 'int64',
            'AQI_Category': 'object',
            'Population_M': 'float64',
            'Vehicle_Density_per_km2': 'int64',
            'Industrial_Activity_Score': 'int64',
            'Forest_Cover_pct': 'float64'
        }
        
        # Check for missing columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert data types
        for col, dtype in expected_columns.items():
            if col in df.columns:
                try:
                    if dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    elif dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to {dtype}: {str(e)}")
        
        # Clean AQI_Category values
        if 'AQI_Category' in df.columns:
            df['AQI_Category'] = df['AQI_Category'].str.strip().str.title()
        
        print(f"Successfully loaded {len(df)} records")
        print(f"Data columns: {', '.join(df.columns)}")
        print(f"Years available: {sorted(df['Year'].unique())}")
        print(f"Cities available: {len(df['City'].unique())}")
        print(f"Sample data:\n{df.head().to_string()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to load data: {str(e)}")

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
    """
    Compare two cities based on their emissions and air quality data.
    Handles city selection, data loading, and template rendering.
    """
    try:
        print("\n=== Starting compare route ===")
        
        # Load data with error handling
        try:
            df = load_data()
            print(f"Successfully loaded data with {len(df)} rows")
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            print(error_msg)
            return render_template('error.html', 
                                error_message=error_msg,
                                error_code=500), 500
        
        # Get unique cities for the dropdown
        try:
            available_cities = sorted(df['City'].astype(str).unique().tolist())
            if not available_cities:
                raise ValueError("No city data available")
                
            print(f"Found {len(available_cities)} available cities")
            
        except Exception as e:
            error_msg = f"Error processing city list: {str(e)}"
            print(error_msg)
            return render_template('error.html',
                                error_message=error_msg,
                                error_code=500), 500
        
        # Get city parameters if they exist
        city1 = request.args.get('city1', '').strip()
        city2 = request.args.get('city2', '').strip()
        print(f"Requested cities - city1: '{city1}', city2: '{city2}'")
        
        # If no cities are selected, use the first two available cities as default
        if not city1 and not city2 and len(available_cities) >= 2:
            city1 = available_cities[0]
            city2 = available_cities[1] if len(available_cities) > 1 else available_cities[0]
            print(f"Using default cities: {city1} and {city2}")
            
            # Redirect to the same URL with default cities to update the URL in the browser
            return redirect(url_for('main.compare', city1=city1, city2=city2))
        
        # If still no cities, return an error
        if not city1 or not city2:
            error_msg = "Please select two different cities to compare."
            print(error_msg)
            return render_template('error.html',
                                error_message=error_msg,
                                error_code=400), 400
        
        # Check if the provided cities exist in the dataset
        if city1 not in available_cities or city2 not in available_cities:
            error_msg = f"One or both cities not found in the dataset. Please select from the available cities."
            print(error_msg)
            return render_template('error.html',
                                error_message=error_msg,
                                error_code=404), 404
        
        # Get city data for the template
        try:
            print(f"Fetching data for cities: {city1} and {city2}")
            
            # Get all data for both cities for trend analysis
            city1_data_all = df[df['City'] == city1].sort_values('Year')
            city2_data_all = df[df['City'] == city2].sort_values('Year')
            
            # Get the most recent data for each city
            city1_latest = city1_data_all.iloc[-1].to_dict()
            city2_latest = city2_data_all.iloc[-1].to_dict()
            
            # Prepare data for charts
            def prepare_city_data(city_name, city_df):
                # Get all years of data for the city
                years = city_df['Year'].astype(int).tolist()
                
                # Prepare emissions data for all available years
                emissions_data = {}
                for year in years:
                    year_data = city_df[city_df['Year'] == year].iloc[0]
                    emissions_data[str(year)] = float(year_data.get('CO2_Emission_kt', 0))
                
                # Get latest data
                latest = city_df.iloc[-1]
                
                return {
                    'name': city_name,
                    'state': latest.get('State', ''),
                    'year': int(latest.get('Year', 2023)),
                    'emissions': emissions_data,
                    'metrics': {
                        'co2': float(latest.get('CO2_Emission_kt', 0)),
                        'so2': float(latest.get('SO2_Annual_Avg_ugm3', 0)),
                        'no2': float(latest.get('NO2_Annual_Avg_ugm3', 0)),
                        'pm10': float(latest.get('PM10_Annual_Avg_ugm3', 0)),
                        'pm25': float(latest.get('PM2.5_Annual_Avg_ugm3', 0)),
                        'aqi': int(latest.get('AQI_Index', 0)),
                        'population': float(latest.get('Population_M', 0)),
                        'vehicle_density': int(latest.get('Vehicle_Density_per_km2', 0)),
                        'industry': float(latest.get('Industrial_Activity_Score', 0)),
                        'forest_cover': float(latest.get('Forest_Cover_pct', 0))
                    }
                }
            
            # Prepare data for both cities
            city1_data = prepare_city_data(city1, city1_data_all)
            city2_data = prepare_city_data(city2, city2_data_all)
            
            # Get unique years for x-axis
            all_years = sorted(list(set(city1_data['emissions'].keys()) | set(city2_data['emissions'].keys())))
            
            # Prepare the final data structure for the template
            template_data = {
                'city1': city1_data['name'],
                'city2': city2_data['name'],
                'city1_data': city1_data,
                'city2_data': city2_data,
                'available_cities': available_cities,
                'visualizations': {}
            }
            
            print(f"Prepared data for template. City1: {city1}, City2: {city2}")
            print(f"City 1 data sample: {str(city1_data)[:200]}...")
            print(f"City 2 data sample: {str(city2_data)[:200]}...")
            
            # Generate comparison visualizations
            try:
                # Create a temporary DataFrame with just the two cities for visualization
                comparison_df = pd.concat([city1_data_all, city2_data_all])
                
                # Create visualizations
                emissions_plot = create_emissions_trend_plot(comparison_df)
                aqi_plot = create_aqi_emissions_plot(comparison_df)
                industry_plot = create_industry_emissions_plot(comparison_df)
                
                # Add visualization URLs to the template data
                template_data['visualizations'] = {
                    'emissions_trend': emissions_plot,
                    'aqi_vs_emissions': aqi_plot,
                    'industry_vs_emissions': industry_plot
                }
                
                print("Successfully generated visualizations")
                
            except Exception as e:
                print(f"Warning: Could not generate all visualizations: {str(e)}")
                # Continue without visualizations if there's an error
                import traceback
                traceback.print_exc()
            
            return render_template('comparison.html', **template_data)
            
        except Exception as e:
            error_msg = f"Error preparing city data: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return render_template('error.html',
                                error_message=error_msg,
                                error_code=500), 500
                                
        
    except Exception as e:
        error_msg = f"Unexpected error in compare route: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return render_template('error.html',
                            error_message="An unexpected error occurred. Please try again later.",
                            error_code=500), 500
        return "An error occurred while processing your request. Please try again later.", 500

def create_heatmap_plot(df):
    """Create a heatmap of CO2 emissions by city."""
    try:
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            values='CO2_Emission_kt',
            index='City',
            columns='Year',
            aggfunc='mean',
            fill_value=0
        )
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        cmap = LinearSegmentedColormap.from_list('emissions', ['green', 'yellow', 'red'])
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            annot=True,
            fmt=".1f",
            linewidths=.5,
            cbar_kws={'label': 'CO₂ Emissions (kt)'}
        )
        
        plt.title('CO₂ Emissions by City and Year', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('City')
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        return ""

@main.route('/')
def index():
    try:
        df = load_data()
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("No data available. The dataset is empty.")
            
        # Check if required columns exist
        required_columns = ['Year', 'CO2_Emission_kt', 'AQI_Index', 'City']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {', '.join(missing_columns)}")
            
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
        
        # Check if we have data for the latest year
        if latest_data.empty:
            raise ValueError(f"No data available for the latest year: {latest_year}")
        
        # Check if we have valid CO2 emission data
        if latest_data['CO2_Emission_kt'].isnull().all() or latest_data['CO2_Emission_kt'].empty:
            raise ValueError("No valid CO2 emission data available")
            
        # Calculate KPIs with error handling
        total_emissions = latest_data['CO2_Emission_kt'].sum()
        avg_aqi = latest_data['AQI_Index'].mean() if not latest_data['AQI_Index'].empty else 0
        
        # Safely get city with max emissions
        max_emission_idx = latest_data['CO2_Emission_kt'].idxmax()
        max_emission_city = latest_data.loc[max_emission_idx] if not pd.isna(max_emission_idx) else None
        
        # Generate plots
        plots = {
            'emissions_trend': create_emissions_trend_plot(df),
            'city_emissions': create_city_emissions_plot(latest_data),
            'aqi_emissions': create_aqi_emissions_plot(latest_data),
            'industry_emissions': create_industry_emissions_plot(latest_data),
            'correlation_heatmap': create_correlation_heatmap(latest_data),
            'emissions_by_city': create_emissions_by_city_pie(latest_data)
        }
        
        # Prepare top city data with fallbacks
        top_city = {
            'name': max_emission_city['City'] if max_emission_city is not None else 'N/A',
            'emission': max_emission_city['CO2_Emission_kt'] if max_emission_city is not None else 0,
            'aqi': max_emission_city.get('AQI_Index', 0) if max_emission_city is not None else 0
        } if max_emission_city is not None else None
        
        return render_template('index.html',
                            total_emissions=total_emissions,
                            avg_aqi=round(avg_aqi, 1) if pd.notna(avg_aqi) else 0,
                            top_city=top_city,
                            latest_year=latest_year,
                            cities_count=df['City'].nunique() if not df.empty else 0,
                            plots=plots)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return str(e), 500
    
def create_correlation_heatmap(df):
    """Create a correlation heatmap for numerical features with enhanced styling."""
    try:
        # Select only numerical columns for correlation
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numerical_cols].corr()
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create heatmap with better styling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            correlation_matrix, 
            mask=mask,  # Only show lower triangle
            cmap=cmap, 
            vmin=-1, vmax=1, center=0,
            square=True,
            ax=ax, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot=True, 
            fmt=".2f",
            annot_kws={"size": 9}
        )
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
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
    """Create emissions trend plot and return URL to saved image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by year and calculate mean emissions
    yearly_avg = df.groupby('Year')['CO2_Emission_kt'].mean()
    
    # Create the plot
    sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, 
                marker='o', linewidth=2.5, color='#10B981', ax=ax)
    
    # Customize the plot
    ax.set_title('Average CO2 Emissions Trend (2017-2024)', fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('CO2 Emissions (kt)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot and return the URL
    return save_plot(fig, 'emissions_trend')

def create_city_emissions_plot(df):
    """Create city emissions plot and return URL to saved image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get latest year data and sort
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].sort_values('CO2_Emission_kt', ascending=False).head(10)
    
    # Create the plot
    sns.barplot(x='CO2_Emission_kt', y='City', data=latest_data, hue='City', palette='viridis', legend=False, ax=ax)
    
    # Customize the plot
    ax.set_title(f'Top 10 Cities by CO2 Emissions ({latest_year})', fontsize=14, pad=20)
    ax.set_xlabel('CO2 Emissions (kt)', fontsize=12)
    ax.set_ylabel('City', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels on the bars
    for i, v in enumerate(latest_data['CO2_Emission_kt']):
        ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=10)
    
    # Save the plot and return the URL
    return save_plot(fig, 'city_emissions')

def create_aqi_emissions_plot(df):
    """Create AQI vs Emissions scatter plot and return URL to saved image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get latest year data
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    # Create the plot
    scatter = ax.scatter(
        x=latest_data['CO2_Emission_kt'],
        y=latest_data['AQI_Index'],
        c=latest_data['CO2_Emission_kt'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('CO2 Emissions (kt)')
    
    # Add trendline
    z = np.polyfit(latest_data['CO2_Emission_kt'], latest_data['AQI_Index'], 1)
    p = np.poly1d(z)
    ax.plot(latest_data['CO2_Emission_kt'], p(latest_data['CO2_Emission_kt']), 
            'r--', alpha=0.7)
    
    # Customize the plot
    ax.set_title(f'AQI vs CO2 Emissions ({latest_year})', fontsize=14, pad=20)
    ax.set_xlabel('CO2 Emissions (kt)', fontsize=12)
    ax.set_ylabel('AQI Index', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot and return the URL
    return save_plot(fig, 'aqi_emissions')

def create_industry_emissions_plot(df):
    """Create industrial activity vs emissions plot and return URL to saved image."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if required columns exist
        required_columns = ['Industrial_Activity_Score', 'CO2_Emission_kt']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Create a placeholder plot with an informative message
            ax.text(0.5, 0.5, 
                   f'Data not available for: {", ".join(missing_columns)}',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=12)
            ax.set_title('Industrial Activity vs CO₂ Emissions', fontsize=14)
            ax.axis('off')
        else:
            # Group by industrial activity and calculate mean emissions
            industry_emissions = df.groupby('Industrial_Activity_Score')['CO2_Emission_kt'].mean().reset_index()
            
            # Create scatter plot
            sns.scatterplot(
                x='Industrial_Activity_Score',
                y='CO2_Emission_kt',
                data=industry_emissions,
                color='#2ecc71',
                s=100,
                alpha=0.7,
                ax=ax
            )
            
            # Add regression line if we have enough data points
            if len(industry_emissions) > 1:
                sns.regplot(
                    x='Industrial_Activity_Score',
                    y='CO2_Emission_kt',
                    data=industry_emissions,
                    scatter=False,
                    color='#e74c3c',
                    line_kws={'linestyle': '--', 'alpha': 0.7},
                    ax=ax
                )
            
            ax.set_title('Industrial Activity vs CO₂ Emissions', fontsize=14)
            ax.set_xlabel('Industrial Activity Score (0-100)')
            ax.set_ylabel('Average CO₂ Emissions (kt)')
            ax.grid(True, alpha=0.3)
        
        # Save the plot and return the URL
        return save_plot(fig, 'industry_emissions')
        
    except Exception as e:
        print(f"Error creating industry emissions plot: {str(e)}")
        return None

def create_emissions_by_city_pie(df):
    """Create a pie chart of emissions by city and return URL to saved image."""
    try:
        # Get top 5 cities by emissions and group the rest as 'Others'
        top_cities = df.groupby('City')['CO2_Emission_kt'].sum().nlargest(5)
        other_emissions = df[~df['City'].isin(top_cities.index)]['CO2_Emission_kt'].sum()
        
        # Create data for the pie chart
        if other_emissions > 0:
            top_cities['Other Cities'] = other_emissions
        
        # Create a color palette
        colors = sns.color_palette('pastel')[0:len(top_cities)]
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            top_cities,
            labels=top_cities.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='w'),
            textprops={'fontsize': 10}
        )
        
        # Draw a circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.7, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.title('CO₂ Emissions by City', fontsize=14)
        
        # Save the plot and return the URL
        return save_plot(fig, 'emissions_by_city')
        
    except Exception as e:
        print(f"Error creating emissions by city pie chart: {str(e)}")
        return None

def get_emissions_by_city_year():
    """Get emissions data by city and year for heatmap."""
    try:
        df = load_data()
        # Pivot to get cities as rows and years as columns
        pivot_df = df.pivot_table(
            index='City',
            columns='Year',
            values='CO2_Emission_kt',
            aggfunc='mean',
            fill_value=0
        )
        
        # Convert to list of dictionaries for JSON response
        data = []
        for city in pivot_df.index:
            city_data = {'city': city}
            for year in pivot_df.columns:
                city_data[str(year)] = round(pivot_df.loc[city, year], 2)
            data.append(city_data)
            
        return {
            'cities': [str(city) for city in pivot_df.index],
            'years': [str(year) for year in pivot_df.columns],
            'data': data
        }
    except Exception as e:
        print(f"Error getting emissions by city and year: {str(e)}")
        return {'error': str(e)}

@main.route('/api/emissions-heatmap')
def emissions_heatmap_data():
    """Return emissions heatmap data."""
    data = get_emissions_by_city_year()
    return jsonify(data)

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

@main.route('/correlation')
def correlation_analysis():
    """Render the correlation analysis page."""
    try:
        df = load_data()
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
        
        # Generate correlation heatmap
        correlation_plot = create_correlation_heatmap(latest_data)
        
        return render_template('correlation_analysis.html', 
                             plots={'correlation_heatmap': correlation_plot},
                             latest_year=latest_year)
    except Exception as e:
        print(f"Error in correlation_analysis: {str(e)}")
        return str(e), 500

@main.route('/api/predict-co2', methods=['POST'])
@main.route('/heatmap')
def show_heatmap():
    """Render the emissions heatmap page."""
    try:
        df = load_data()
        # Get the most recent year's data
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
        
        # Create a pivot table for the heatmap
        heatmap_data = latest_data.pivot_table(
            values='CO2_Emission_kt',
            index='City',
            columns='Year',
            aggfunc='mean'
        )
        
        # Create the heatmap figure
        plt.figure(figsize=(12, 8))
        
        # Define custom colormap (green-yellow-red)
        cmap = LinearSegmentedColormap.from_list('emissions', ['green', 'yellow', 'red'])
        
        # Create the heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            linewidths=.5,
            cbar_kws={'label': 'CO₂ Emissions (kt)'}
        )
        
        plt.title(f'CO₂ Emissions by City ({latest_year})', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('City')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return render_template('heatmap.html', 
                            plot_url=f'data:image/png;base64,{plot_data}',
                            year=latest_year)
    
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return render_template('error.html', message="Error generating heatmap")


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
