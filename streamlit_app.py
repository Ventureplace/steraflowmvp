import streamlit as st
import pandas as pd
from prep import clean_dataframe, initialize_openai_client
import os

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Default Project"

# Initialize OpenAI client
initialize_openai_client()

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

    # Data cleaning options
    st.subheader("Data Cleaning Options")

    remove_duplicates = st.checkbox("Remove Duplicate Rows")
    handle_missing = st.checkbox("Handle Missing Values")
    handle_outliers = st.checkbox("Handle Outliers")
    normalize_data = st.checkbox("Normalize Data")
    
    variance_threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.1, 0.01)
    skew_threshold = st.slider("Skew Threshold", 0.0, 1.0, 0.5, 0.01)

    # Apply cleaning when button is clicked
    if st.button("Apply Data Cleaning"):
        cleaned_data = clean_dataframe(
            data,
            remove_duplicates,
            handle_missing,
            handle_outliers,
            normalize_data,
            variance_threshold,
            skew_threshold
        )

        # Save cleaned data
        st.session_state.projects[project_name]['cleaned_data'] = cleaned_data
        st.success("Data cleaning applied and saved successfully!")

    # Display cleaned data
    if 'cleaned_data' in st.session_state.projects[project_name]:
        st.subheader("Cleaned Data Sample")
        st.write(st.session_state.projects[project_name]['cleaned_data'].head())
else:
    st.warning("No data available. Please upload data first.")
