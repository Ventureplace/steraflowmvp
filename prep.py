import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from openai import OpenAI
import json
from io import BytesIO

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
    try:
        api_key = st.secrets["openai_api_key"]
    except FileNotFoundError:
        st.warning("No secrets file found. You'll need to input your OpenAI API key manually.")
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.error("Please enter a valid OpenAI API key to use the AI-powered SQL generation feature.")

def clean_dataframe(df, remove_duplicates, handle_missing, handle_outliers, normalize_data, variance_threshold, skew_threshold):
    if DATAMANCER_AVAILABLE:
        cleaned_df = smart_clean_extended(
            df,
            variance_threshold=variance_threshold,
            skew_threshold=skew_threshold
        )
    else:
        cleaned_df = df.copy()
        st.warning("Advanced cleaning functions are not available. Performing basic cleaning only.")

    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if handle_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    if handle_outliers:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            cleaned_df[col] = cleaned_df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    if normalize_data:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = (cleaned_df[col] - cleaned_df[col].min()) / (cleaned_df[col].max() - cleaned_df[col].min())

    return cleaned_df

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

    # Data cleaning options
    st.subheader("Data Cleaning Options")

    # Handle missing values
    if st.checkbox("Handle Missing Values"):
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                method = st.selectbox(f"Handle missing values in {column}", 
                                      ["Drop", "Fill with mean", "Fill with median", "Fill with mode"])
                if method == "Drop":
                    data = data.dropna(subset=[column])
                elif method == "Fill with mean":
                    data[column].fillna(data[column].mean(), inplace=True)
                elif method == "Fill with median":
                    data[column].fillna(data[column].median(), inplace=True)
                elif method == "Fill with mode":
                    data[column].fillna(data[column].mode()[0], inplace=True)

    # Remove duplicates
    if st.checkbox("Remove Duplicate Rows"):
        data = data.drop_duplicates()

    # Handle outliers
    if st.checkbox("Handle Outliers"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if st.checkbox(f"Handle outliers in {column}"):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[column] = data[column].clip(lower_bound, upper_bound)

    # Save cleaned data
    if st.button("Save Cleaned Data"):
        st.session_state.projects[project_name]['cleaned_data'] = data
        st.success("Cleaned data saved successfully!")

    st.subheader("Cleaned Data Sample")
    st.write(data.head())

if __name__ == "__main__":
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'current_project' not in st.session_state:
        st.session_state.current_project = "Default Project"
    show(st.session_state.current_project)