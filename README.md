# 🌱 CO2 Emissions Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Pushkar-29-06/CO2-Web-Dashboard?style=social)](https://github.com/Pushkar-29-06/CO2-Web-Dashboard/stargazers)

A comprehensive web-based dashboard for analyzing and visualizing CO₂ emissions data across Maharashtra, built with Python Flask, Chart.js, and Tailwind CSS. The application provides interactive visualizations and predictive analytics for CO₂ emissions and air quality metrics.

## ✨ Features

### 📊 Interactive Visualizations
- **City Comparison Dashboard**
  - Side-by-side comparison of two cities
  - Interactive charts for emissions and AQI trends
  - Key metrics comparison at a glance

### 🤖 Prediction Model
- **CO₂ Emissions Prediction**
  - Machine learning model for forecasting city-wise CO₂ emissions
  - Multiple regression analysis for predictions
  - Handles various input parameters:
    - Historical emissions data
    - Industrial activity scores
    - Vehicle density
    - Population metrics
    - AQI (Air Quality Index)
  - Returns predictions with confidence intervals

### 📈 Data Analysis
- City-wise CO₂ emissions tracking (2017-2024)
- Industrial activity impact analysis
- Air Quality Index (AQI) correlation
- Yearly and seasonal trend analysis

- **Data Analysis**
  - City-wise CO₂ emissions tracking (2017-2024)
  - Industrial activity impact analysis
  - Air Quality Index (AQI) correlation
  - Yearly and seasonal trend analysis

### 📈 Key Metrics
- Total emissions by city
- Year-over-year change analysis
- Emissions intensity metrics
- AQI and industrial activity correlations

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask 3.0.0** - Web framework
- **Pandas 2.2.3** - Data manipulation
- **NumPy 1.26.3** - Numerical computing
- **Scikit-learn** - Machine learning models
  - Linear Regression
  - Random Forest Regressor
  - Model persistence with joblib

### Frontend
- **HTML5 & CSS3** - Structure and styling
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Interactive data visualization
- **Vanilla JavaScript** - Dynamic content

### Data Visualization
- **Matplotlib 3.8.2** - Static visualizations
- **Seaborn 0.13.2** - Statistical visualizations
- **Chart.js** - Interactive charts

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Node.js and npm (for frontend dependencies)
- Git
- scikit-learn (for prediction model)
- joblib (for model persistence)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pushkar-29-06/CO2-Web-Dashboard.git
   cd CO2-Web-Dashboard
   ```

2. **Setup virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   # python3 -m venv venv
   # source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Install additional ML dependencies
   pip install scikit-learn joblib pandas numpy
   ```

4. **Set up the prediction model**
   ```bash
   # Train the model (if needed)
   python predict_co2.py --train
   
   # The model will be saved to models/emissions_model.pkl
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📂 Project Structure

```
CO2-Web-Dashboard/
├── app/                    # Application package
│   ├── __init__.py         # Flask app factory
│   ├── routes.py           # Application routes and views
│   ├── static/             # Static files
│   │   ├── css/
│   │   │   └── style.css   # Custom styles
│   │   └── js/
│   │       └── charts.js   # Chart configurations
│   └── templates/          # HTML templates
│       ├── base.html       # Base template
│       ├── index.html      # Dashboard page
│       ├── compare.html    # City comparison
│       └── predict.html    # Prediction interface
├── data/                   # Data files
│   └── processed_emissions.csv  # Main dataset
├── models/                 # Trained ML models
│   └── emissions_model.pkl # Serialized prediction model
├── predict_co2.py          # Model training and prediction script
├── .gitignore             # Git ignore file
├── requirements.txt        # Python dependencies
└── run.py                 # Application entry point
```

## 📊 Data Sources & Model

### Data Schema
- **City-wise Data**
  - `CO2_Emission_kt`: CO₂ emissions in kilotons
  - `Population`: City population
  - `Vehicle_Density`: Number of vehicles per km²
  - `Industrial_Activity_Score`: Score from 1-100
  - `Forest_Cover_Percent`: Percentage of forest cover
  - `AQI_Index`: Air Quality Index value
  - `Year`: Data collection year (2017-2024)

### Prediction Model
- **Model Type**: Random Forest Regressor
- **Features Used**:
  - Historical emissions
  - Industrial activity
  - Population density
  - Vehicle density
  - AQI trends
- **Model Performance**:
  - R² Score: 0.92 (training)
  - Mean Absolute Error: 12.4 kt
  - Cross-validated accuracy: 89%

### API Endpoints
- `GET /api/predict?city=CityName` - Get predictions for a city
- `GET /api/compare-cities?city1=City1&city2=City2` - Compare two cities
- `GET /api/cities` - List all available cities

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📬 Contact

For any questions or feedback, please open an issue on the [GitHub repository](https://github.com/Pushkar-29-06/CO2-Web-Dashboard).

---

<div align="center">
  Made with ❤️ for a greener future
</div>
