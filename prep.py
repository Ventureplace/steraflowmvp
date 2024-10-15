import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from openai import OpenAI
import json
from io import BytesIO
from scipy import stats

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from datamancer.cleaner import smart_clean_extended
    from datamancer.insights import generate_data_report, plot_data_insights
    from datamancer.type_infer import infer_types, TypeInformation
    DATAMANCER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing from datamancer. {str(e)}")
    DATAMANCER_AVAILABLE = False

# Initialize OpenAI client
client = None

def initialize_openai_client():
    global client
    api_key = None

    # Try to get the API key from Streamlit secrets
    try:
        api_key = st.secrets["openai_api_key"]
    except KeyError:
        st.warning("OpenAI API key not found in Streamlit secrets.")

    # If API key is not in secrets, prompt the user
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            st.success("API key entered successfully!")
        else:
            st.error("Please enter a valid OpenAI API key to use the AI-powered features.")

    # Initialize the client if we have an API key
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None

    return client

def clean_dataframe(df, options):
    if not isinstance(df, pd.DataFrame):
        st.error(f"Input data is not a pandas DataFrame. Type: {type(df)}")
        return None

    cleaned_df = df.copy()

    try:
        if options['remove_duplicates']:
            cleaned_df = cleaned_df.drop_duplicates()

        if options['handle_missing']:
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        if options['missing_numeric_method'] == 'mean':
                            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        elif options['missing_numeric_method'] == 'median':
                            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                        else:  # mode
                            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)

        if options['handle_outliers']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if options['outlier_method'] == 'IQR':
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                elif options['outlier_method'] == 'zscore':
                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                    cleaned_df[col] = cleaned_df[col].mask(z_scores > 3, cleaned_df[col].median())

        if options['normalize_data']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if options['scaling_method'] == 'minmax':
                    min_val = cleaned_df[col].min()
                    max_val = cleaned_df[col].max()
                    cleaned_df[col] = (cleaned_df[col] - min_val) / (max_val - min_val)
                else:  # standard
                    mean_val = cleaned_df[col].mean()
                    std_val = cleaned_df[col].std()
                    cleaned_df[col] = (cleaned_df[col] - mean_val) / std_val

        if options['remove_low_variance']:
            variance = cleaned_df.var()
            cleaned_df = cleaned_df.loc[:, variance > options['variance_threshold']]

        if options['handle_skewness']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if abs(stats.skew(cleaned_df[col])) > options['skew_threshold']:
                    cleaned_df[col] = np.log1p(cleaned_df[col] - cleaned_df[col].min() + 1)

        return cleaned_df

    except Exception as e:
        st.error(f"An error occurred during data cleaning: {str(e)}")
        return None

def show(project_name):
    st.title(f"Data Cleaning for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    if 'data' not in st.session_state.projects[project_name] or st.session_state.projects[project_name]['data'] is None:
        st.warning("No data available. Please upload data first.")
        return

    data = st.session_state.projects[project_name]['data']

    if not isinstance(data, pd.DataFrame):
        st.error("The data is not in the correct format. Please upload a valid dataset.")
        return

    st.subheader("Original Data Sample")
    st.write(data.head())

    st.subheader("Data Cleaning Options")

    options = {
        'remove_duplicates': st.checkbox("Remove Duplicate Rows"),
        'handle_missing': st.checkbox("Handle Missing Values"),
        'missing_numeric_method': st.selectbox("Missing Numeric Values Method", ['mean', 'median', 'mode']),
        'handle_outliers': st.checkbox("Handle Outliers"),
        'outlier_method': st.selectbox("Outlier Handling Method", ['IQR', 'zscore']),
        'normalize_data': st.checkbox("Normalize/Scale Data"),
        'scaling_method': st.selectbox("Scaling Method", ['minmax', 'standard']),
        'remove_low_variance': st.checkbox("Remove Low Variance Features"),
        'variance_threshold': st.slider("Variance Threshold", 0.0, 1.0, 0.1, 0.01),
        'handle_skewness': st.checkbox("Handle Skewness"),
        'skew_threshold': st.slider("Skewness Threshold", 0.0, 1.0, 0.5, 0.01)
    }

    if st.button("Apply Data Cleaning"):
        cleaned_data = clean_dataframe(data, options)
        if cleaned_data is not None:
            st.session_state.projects[project_name]['cleaned_data'] = cleaned_data
            st.success("Data cleaning applied and saved successfully!")

            st.subheader("Cleaned Data Sample")
            st.write(cleaned_data.head())

            st.subheader("Cleaning Summary")
            st.write(f"Original shape: {data.shape}")
            st.write(f"Cleaned shape: {cleaned_data.shape}")
            
            if options['remove_duplicates']:
                st.write(f"Duplicates removed: {data.shape[0] - cleaned_data.shape[0]}")
            
            if options['handle_missing']:
                st.write("Missing values handled")
            
            if options['handle_outliers']:
                st.write(f"Outliers handled using {options['outlier_method']} method")
            
            if options['normalize_data']:
                st.write(f"Data normalized using {options['scaling_method']} scaling")
            
            if options['remove_low_variance']:
                st.write(f"Low variance features removed (threshold: {options['variance_threshold']})")
            
            if options['handle_skewness']:
                st.write(f"Skewness handled (threshold: {options['skew_threshold']})")
        else:
            st.error("Data cleaning failed. Please check your data and try again.")

if __name__ == "__main__":
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'current_project' not in st.session_state:
        st.session_state.current_project = "Default Project"
    show(st.session_state.current_project)
