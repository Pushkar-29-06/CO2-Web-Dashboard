from flask import Blueprint, render_template, jsonify, send_file
import pandas as pd
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
plt.style.use('ggplot')  # Using 'ggplot' style which is similar to seaborn
sns.set_theme(style="whitegrid")

main = Blueprint('main', __name__)

# Load data
def load_data():
    data_path = os.path.join('data', 'processed_emissions.csv')
    return pd.read_csv(data_path)

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
    plt.figure(figsize=(10, 6))
    
    # Get latest year data
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    # Create the plot
    sns.regplot(
        x='Industrial_Activity_Score',
        y='CO2_Emission_kt',
        data=latest_data,
        scatter_kws={'s': 100, 'alpha': 0.6, 'edgecolor': 'w', 'linewidths': 0.5},
        line_kws={'color': 'red', 'linestyle': '--'}
    )
    
    # Customize the plot
    plt.title(f'Industrial Activity vs CO2 Emissions ({latest_year})', fontsize=14, pad=20)
    plt.xlabel('Industrial Activity Score', fontsize=12)
    plt.ylabel('CO2 Emissions (kt)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation coefficient
    corr = latest_data['Industrial_Activity_Score'].corr(latest_data['CO2_Emission_kt'])
    plt.annotate(f'Correlation: {corr:.2f}', 
                xy=(0.7, 0.9), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')
