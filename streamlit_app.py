import streamlit as st
import pandas as pd
from prep import load_config, handle_missing_values, handle_outliers, normalize_data, detect_anomalies, create_correlation_plots, feature_engineering
import os

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Default Project"

st.title("🎈 Data Cleaning App")

# Project selection
project_name = st.text_input("Project Name", value=st.session_state.current_project)
st.session_state.current_project = project_name

if project_name not in st.session_state.projects:
    st.session_state.projects[project_name] = {'data': None, 'cleaned_data': None}

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state.projects[project_name]['data'] = data
    st.write("Data uploaded successfully!")

# Main app logic
if st.session_state.projects[project_name]['data'] is not None:
    data = st.session_state.projects[project_name]['data']

    st.subheader("Original Data Sample")
    st.write(data.head())

    # Load configuration
    config = load_config()

    # Data cleaning options
    st.subheader("Data Cleaning Options")

    # ... (include all the options from the original show() function)

    # Apply cleaning when button is clicked
    if st.button("Apply Data Cleaning"):
        # ... (include all the cleaning steps from the original show() function)

        # Save cleaned data
        st.session_state.projects[project_name]['cleaned_data'] = data
        st.success("Data cleaning applied and saved successfully!")

        # Generate plots
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        create_correlation_plots(data, output_dir)
        st.success("Correlation and missing values plots generated.")

    # Display cleaned data
    if 'cleaned_data' in st.session_state.projects[project_name]:
        st.subheader("Cleaned Data Sample")
        st.write(st.session_state.projects[project_name]['cleaned_data'].head())
else:
    st.warning("No data available. Please upload data first.")
