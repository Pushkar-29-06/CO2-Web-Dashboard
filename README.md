# 🌱 CO2 Emissions Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive web-based dashboard for analyzing and visualizing CO₂ emissions data across Maharashtra, built with Python Flask and Matplotlib.

## ✨ Features

- **Interactive Visualizations**
  - Line charts for trend analysis
  - Bar charts for city-wise comparisons
  - Scatter plots for correlation analysis
  - Responsive design for all devices

- **Data Analysis**
  - City-wise CO₂ emissions tracking
  - Yearly trends (2017-2024)
  - Industrial activity impact analysis
  - Air Quality Index (AQI) correlation

- **Key Metrics**
  - Total emissions by city
  - Year-over-year change
  - Emissions per capita
  - Sector-wise breakdown

## 🛠️ Tech Stack

- **Backend**
  - Python 3.8+
  - Flask 3.0.0
  - Pandas 2.2.3
  - NumPy 1.26.3

- **Visualization**
  - Matplotlib 3.8.2
  - Seaborn 0.13.2

- **Frontend**
  - HTML5
  - CSS3 (Tailwind CSS)
  - Vanilla JavaScript

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

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
│   │   └── css/
│   │       └── style.css   # Custom styles
│   └── templates/          # HTML templates
│       ├── base.html       # Base template
│       └── index.html      # Dashboard page
├── data/                   # Data files
│   └── processed_emissions.csv  # Main dataset
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
└── run.py                # Application entry point
```

## 📊 Data Sources

The dashboard uses processed CO₂ emissions data with the following metrics:
- **City-wise Data**
  - CO₂ emissions (kt)
  - Population
  - Vehicle density
  - Industrial activity score
  - Forest cover (%)
  - AQI (Air Quality Index)

- **Time Series**
  - Annual data from 2017 to 2024
  - Seasonal variations
  - Growth trends

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
