import streamlit as st
from prep import initialize_openai_client, clean_dataframe
import dashboard
import pandas as pd

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Default Project"

# Initialize OpenAI client
initialize_openai_client()

st.title("🎈 Data Analysis App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Clean Data", "Dashboard"])

# Project selection
project_name = st.sidebar.text_input("Project Name", value=st.session_state.current_project)
st.session_state.current_project = project_name

if project_name not in st.session_state.projects:
    st.session_state.projects[project_name] = {'data': None, 'cleaned_data': None}

if page == "Upload Data":
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.projects[project_name]['data'] = data
        st.write("Data uploaded successfully!")
        st.write(data.head())

elif page == "Clean Data":
    st.header("Clean Data")
    if st.session_state.projects[project_name]['data'] is not None:
        data = st.session_state.projects[project_name]['data']
        
        # Data cleaning options
        remove_duplicates = st.checkbox("Remove Duplicate Rows")
        handle_missing = st.checkbox("Handle Missing Values")
        handle_outliers = st.checkbox("Handle Outliers")
        normalize_data = st.checkbox("Normalize Data")
        
        variance_threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.1, 0.01)
        skew_threshold = st.slider("Skew Threshold", 0.0, 1.0, 0.5, 0.01)

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
            st.session_state.projects[project_name]['cleaned_data'] = cleaned_data
            st.success("Data cleaning applied and saved successfully!")
    else:
        st.warning("No data available. Please upload data first.")

elif page == "Dashboard":
    dashboard.show(project_name)

# Check OpenAI client status
if 'client' in globals() and globals()['client'] is not None:
    st.sidebar.success("OpenAI client initialized")
else:
    st.sidebar.warning("OpenAI client not initialized. Some features may be limited.")
