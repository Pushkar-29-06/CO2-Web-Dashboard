import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data(filepath):
    """Load and preprocess the emissions data."""
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Convert Year to numeric (in case it's read as string)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Handle any missing values (fill with median for numerical columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def prepare_features_targets(df):
    """Prepare features (X) and target (y) for the model."""
    # Features to use for prediction
    features = [
        'Year',
        'Population_M',
        'Vehicle_Density_per_km2',
        'Industrial_Activity_Score',
        'Forest_Cover_pct',
        'SO2_Annual_Avg_ugm3',
        'NO2_Annual_Avg_ugm3',
        'PM10_Annual_Avg_ugm3',
        'PM2.5_Annual_Avg_ugm3',
        'AQI_Index'
    ]
    
    # Target variable
    target = 'CO2_Emission_kt'
    
    # Get features and target
    X = df[features]
    y = df[target]
    
    return X, y

def train_model(X, y):
    """Train a Random Forest regression model."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize the model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return model, X_test, y_test, y_pred, mse, r2

def save_model(model, model_dir='models'):
    """Save the trained model to disk."""
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'co2_emission_predictor.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def predict_emission(model, input_data):
    """Make a prediction using the trained model."""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

def main():
    # File path to the data
    data_file = 'data/processed_emissions.csv'
    
    try:
        # Load and preprocess data
        print(f"Loading data from {data_file}...")
        df = load_and_preprocess_data(data_file)
        
        # Prepare features and target
        X, y = prepare_features_targets(df)
        
        # Train the model
        print("Training the model...")
        model, X_test, y_test, y_pred, mse, r2 = train_model(X, y)
        
        # Save the model
        model_path = save_model(model)
        
        # Example prediction
        example_input = {
            'Year': 2025,
            'Population_M': 2.5,
            'Vehicle_Density_per_km2': 500,
            'Industrial_Activity_Score': 75,
            'Forest_Cover_pct': 15.0,
            'SO2_Annual_Avg_ugm3': 35.0,
            'NO2_Annual_Avg_ugm3': 40.0,
            'PM10_Annual_Avg_ugm3': 90.0,
            'PM2.5_Annual_Avg_ugm3': 60.0,
            'AQI_Index': 65
        }
        
        prediction = predict_emission(model, example_input)
        print(f"\nExample prediction for input: {example_input}")
        print(f"Predicted CO2 Emissions: {prediction:.2f} kt")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
