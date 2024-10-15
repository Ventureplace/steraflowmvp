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
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)
        st.success("API key entered successfully!")
    else:
        client = None
        st.error("Please enter a valid OpenAI API key to use the AI-powered features.")
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

def get_ai_cleaning_suggestions(data):
    global client
    try:
        if client is None:
            st.error("OpenAI client is not initialized. Please enter your API key.")
            return "Unable to get AI cleaning suggestions at this time."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful data cleaning assistant."},
                {"role": "user", "content": f"Given the following data:\n\n{data.describe().to_string()}\n\nProvide suggestions for data cleaning:"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting AI cleaning suggestions: {str(e)}")
        return "Unable to get AI cleaning suggestions at this time."

def modify_data_with_ai(prompt, data):
    global client
    if client is None:
        st.error("OpenAI client is not initialized. Please enter your API key.")
        return "Unable to modify data with AI at this time."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful data modification assistant. Provide Python code to modify the data based on the user's request."},
            {"role": "user", "content": f"Given the following data:\n\n{data.to_string()}\n\nUser request: {prompt}\n\nProvide Python code to modify the data:"}
        ]
    )
    return response.choices[0].message.content

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

    st.subheader("Original Data")
    st.dataframe(data)  # Display the original data as a dataframe
    
    st.subheader("Edit Data")
    edited_data = st.data_editor(data, num_rows="dynamic", key=f"editor_{project_name}_original")
    
    if not edited_data.equals(data):
        st.session_state.projects[project_name]['data'] = edited_data
        st.success("Data updated successfully!")
    
    data = edited_data  # Use the edited data for further processing

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

            st.subheader("Cleaned Data")
            st.dataframe(cleaned_data)  # Display the cleaned data as a dataframe
            
            st.subheader("Edit Cleaned Data")
            final_cleaned_data = st.data_editor(cleaned_data, num_rows="dynamic", key=f"editor_{project_name}_cleaned")
            
            if not final_cleaned_data.equals(cleaned_data):
                st.session_state.projects[project_name]['cleaned_data'] = final_cleaned_data
                st.success("Cleaned data updated successfully!")
            
            cleaned_data = final_cleaned_data  # Use the final edited cleaned data

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

    st.subheader("AI Cleaning Suggestions")
    if st.button("Get AI Cleaning Suggestions"):
        with st.spinner("Getting AI suggestions..."):
            suggestions = get_ai_cleaning_suggestions(data)
            st.write(suggestions)

    st.subheader("Modify Data with AI")
    modification_prompt = st.text_input("Describe how you want to modify the data:")
    if modification_prompt:
        modification_code = modify_data_with_ai(modification_prompt, data)
        st.code(modification_code, language="python")
        if st.button("Apply Modification"):
            try:
                exec(modification_code)
                st.session_state.projects[project_name]['data'] = data
                st.success("Data modified successfully!")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error modifying data: {str(e)}")

if __name__ == "__main__":
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'current_project' not in st.session_state:
        st.session_state.current_project = "Default Project"
    show(st.session_state.current_project)
