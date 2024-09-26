import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Tea Yield Optimization and Recommendation System", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background-color: #FFFFFF;  /* White background */
        color: #E8F5E9;  /* Dark grey text */
    }
    .title {
        font-size: 40px;
        text-align: left;
        color: #6B8E23;  /* Green color for title */
    }
    .header {
        background-color: #E8F5E9;  /* Light green background */
        padding: 100px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        margin-top: 20px;
    }
    /* Navigation bar styling */
    .css-1y4y4s8 {
        background-color: #6B8E23; /* Sidebar color */
    }
    button {
        background-color: #000000; /* Green button color */
        color: white;
    }
    button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.image("C:/Users/Dell/Downloads/Green_Cute_Tea_Plantation_Text_Logo-removebg-preview.png", width=180)
st.sidebar.title("")

option = st.sidebar.selectbox("Choose an option:", ["Home", "Upload Data", "Train Model and Prediction", "Prediction", "Recommendations", "Additional Recommendations"])

# Welcome (Home) section
if option == "Home":
    st.markdown("<h1 class='title'>Tea Yield Optimization and Recommendation System</h1>", unsafe_allow_html=True)
    st.write("This system helps tea plantation owners optimize their tea yield by analyzing various factors such as soil condition, weather patterns, irrigation practices, and more.")
    st.write("### Instructions:")
    st.write("1. Use the sidebar to navigate through different functionalities.")
    st.write("2. Upload Data: Upload your CSV dataset containing tea yield factors.")
    st.write("3. Train Model and Prediction: Train a predictive model using your uploaded data.")
    st.write("4. Prediction: Enter details about your tea field to predict the yield.")
    st.write("5. Recommendations: View general recommendations for improving tea yield.")
    st.write("6. Additional Recommendations: Get personalized recommendations based on specific soil and nutrient inputs.")

# Upload Data section
if option == "Upload Data":
    st.title("Upload Your Tea Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file, encoding='latin1')
            st.write("File uploaded successfully!")
            
            # Display the columns of the dataset
            st.write("Columns in the uploaded file:")
            st.write(list(data.columns))
            
            st.write("First few rows of the data:")
            st.dataframe(data.head())

            # Drop unnecessary columns (e.g., unnamed columns)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

            # Rename any incorrect columns
            data.rename(columns={'Yeild Quantity (kg)': 'Yield Quantity (kg)'}, inplace=True)

            # Define required columns
            required_columns = ['soil_ph', 'nitrogen_percent', 'phosphorus_percent', 'potassium_percent', 
                                'rainfall_mm', 'temperature_celsius', 'fertilizer_type', 
                                'fertilizer_quantity_kg', 'harvest_date', 'yield_quantity_kg']

            # Check if all required columns are present
            if all(col in data.columns for col in required_columns):
                st.success("All required columns are present!")
                
                # Saving the dataset in session state for later use
                st.session_state['data'] = data
                
                # Extract unique fertilizer types and save them in session state
                unique_fertilizer_types = data['fertilizer_type'].unique().tolist()
                st.session_state['fertilizer_types'] = unique_fertilizer_types
            else:
                st.error("The uploaded CSV file is missing one or more required columns.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.write("Please upload a CSV file.")

# Train Model and Prediction section
if option == "Train Model and Prediction" and 'data' in st.session_state:
    st.title("Train Yield Prediction Model")
    data = st.session_state['data']
    
    # Preprocessing
    st.write("Preprocessing the data...")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    st.write("Missing values in each column:")
    st.write(missing_values[missing_values > 0])

    # Remove rows with missing target values
    data = data.dropna(subset=['yield_quantity_kg'])

    # Label encode categorical variables
    label_encoder = LabelEncoder()
    data['fertilizer_type'] = label_encoder.fit_transform(data['fertilizer_type'])
    
    # Saving label encoder to session state
    st.session_state['label_encoder'] = label_encoder
    
    # Selecting features and target
    X = data[['soil_ph', 'nitrogen_percent', 'phosphorus_percent', 'potassium_percent', 
              'rainfall_mm', 'temperature_celsius', 'fertilizer_type', 'fertilizer_quantity_kg']]
    y = data['yield_quantity_kg']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model (Random Forest in this case)
    st.write("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    st.write("Evaluating the model...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r_squared = r2_score(y_test, y_pred)  # R-squared

    # Display the metrics
    st.write(f"Mean Squared Error on test set: {mse:.2f}")
    st.write(f"Root Mean Squared Error on test set: {rmse:.2f}")
    st.write(f"R-squared on test set: {r_squared:.2f}")

    # Plotting predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Identity line
    plt.title('Predicted vs Actual Yield')
    plt.xlabel('Actual Yield (kg)')
    plt.ylabel('Predicted Yield (kg)')
    plt.grid()
    st.pyplot(plt)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Average Cross-Validation Score: {cv_scores.mean():.2f}")

    st.session_state['model'] = model
    st.success("Model trained successfully!")

# Prediction section
if option == "Prediction" and 'model' in st.session_state:
    st.title("Tea Yield Prediction")
    
    # Input fields for prediction
    st.write("Enter the details for your tea field:")
    
    soil_ph = st.number_input("Soil pH", value=5.5, min_value=3.0, max_value=8.0, step=0.1)
    nitrogen = st.number_input("Nitrogen (%)", value=2.0, min_value=0.0, max_value=10.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (%)", value=0.5, min_value=0.0, max_value=5.0, step=0.1)
    potassium = st.number_input("Potassium (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", value=100.0, min_value=0.0, max_value=1000.0, step=10.0)
    temperature = st.number_input("Temperature (°C)", value=25.0, min_value=10.0, max_value=40.0, step=1.0)
    
    # Retrieve the label encoder from session state
    label_encoder = st.session_state['label_encoder']
    fertilizer_type = st.selectbox("Fertilizer Type", options=label_encoder.classes_)
    fertilizer_qty = st.number_input("Fertilizer Quantity (kg)", value=10.0, min_value=0.0, max_value=1000.0, step=1.0)

    if st.button("Predict Yield"):
        # Prepare input for prediction
        fertilizer_type_encoded = label_encoder.transform([fertilizer_type])[0]
        input_data = np.array([[soil_ph, nitrogen, phosphorus, potassium, rainfall, temperature, fertilizer_type_encoded, fertilizer_qty]])
        
        # Make prediction
        model = st.session_state['model']
        predicted_yield = model.predict(input_data)[0]
        
        st.success(f"Predicted Yield: {predicted_yield:.2f} kg")

# Recommendations section
if option == "Recommendations":
    st.title("General Recommendations for Improving Tea Yield")
    st.write("Based on common practices, here are some recommendations:")
    recommendations = [
        "1. Ensure optimal soil pH (5.5 - 6.5) for better nutrient absorption.",
        "2. Use organic fertilizers to improve soil health.",
        "3. Implement proper irrigation techniques to maintain soil moisture.",
        "4. Monitor weather patterns for optimal planting and harvesting times.",
        "5. Practice crop rotation to maintain soil fertility."
    ]
    for rec in recommendations:
        st.write(rec)

# Additional Recommendations section
if option == "Additional Recommendations":
    st.title("Additional Recommendations Based on Soil and Nutrient Inputs")
    
    # Input fields for personalized recommendations
    st.write("Enter soil and nutrient conditions:")
    
    soil_ph = st.number_input("Soil pH", value=5.5, min_value=3.0, max_value=8.0, step=0.1)
    nitrogen = st.number_input("Nitrogen (%)", value=2.0, min_value=0.0, max_value=10.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (%)", value=0.5, min_value=0.0, max_value=5.0, step=0.1)
    potassium = st.number_input("Potassium (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", value=100.0, min_value=0.0, max_value=1000.0, step=10.0)
    temperature = st.number_input("Temperature (°C)", value=25.0, min_value=10.0, max_value=40.0, step=1.0)

    # Add dropdown for fertilizer type based on uploaded data
    if 'fertilizer_types' in st.session_state:
        fertilizer_type = st.selectbox("Select Fertilizer Type", options=st.session_state['fertilizer_types'])
    else:
        fertilizer_type = st.selectbox("Select Fertilizer Type", options=["No fertilizer types available"])

    if st.button("Get Recommendations"):
        recommendations = []
        
        # Soil pH recommendations
        if soil_ph < 5.5:
            recommendations.append("Increase soil pH using lime.")
        elif soil_ph > 6.5:
            recommendations.append("Lower soil pH by applying sulfur or organic matter.")
        
        # Nutrient recommendations
        if nitrogen < 2.0:
            recommendations.append("Increase nitrogen levels through organic fertilizers or green manure.")
        
        if phosphorus < 0.5:
            recommendations.append("Add phosphorus through bone meal or rock phosphate.")
        
        if potassium < 1.0:
            recommendations.append("Include potassium sources like potash or wood ash.")
        
        if rainfall < 50.0:
            recommendations.append("Increase irrigation to ensure adequate water supply for tea plants.")
        
        if temperature < 20.0:
            recommendations.append("Consider using shade nets to protect plants from cold temperatures.")
        elif temperature > 30.0:
            recommendations.append("Ensure adequate watering and mulch to reduce temperature stress on plants.")

        # Fertilizer type specific recommendations (customize these as needed)
        if fertilizer_type:
            recommendations.append(f"Consider using {fertilizer_type} based on your previous data.")

        if not recommendations:
            st.success("Soil and nutrient levels are adequate for tea cultivation!")
        else:
            st.warning("Recommendations based on your inputs:")
            for rec in recommendations:
                st.write(f"- {rec}")

# Footer
st.markdown("<footer class='footer'>Developed by Tea Yield Optimization Team</footer>", unsafe_allow_html=True)
