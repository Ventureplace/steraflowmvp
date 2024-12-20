import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from openai import OpenAI
import json
from io import BytesIO
from scipy import stats
import utils
import base64
import time

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Initialize session
utils.init()

try:
    from datamancer.cleaner import smart_clean_extended
    from datamancer.insights import generate_data_report, plot_data_insights
    from datamancer.type_infer import infer_types, TypeInformation
    DATAMANCER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing from datamancer. {str(e)}")
    DATAMANCER_AVAILABLE = False

def get_openai_client():
    return OpenAI(api_key=st.secrets["openai_api_key"])

def clean_dataframe(df, options):
    if not isinstance(df, pd.DataFrame):
        st.error(f"Input data is not a pandas DataFrame. Type: {type(df)}")
        return None

    cleaned_df = df.copy()

    try:
        if options['remove_duplicates']:
            cleaned_df = remove_duplicates(cleaned_df)

        if options['handle_missing']:
            cleaned_df = handle_missing(cleaned_df, options['missing_numeric_method'])

        if options['handle_outliers']:
            cleaned_df = handle_outliers(cleaned_df, options['outlier_method'])

        if options['normalize_data']:
            cleaned_df = normalize_data(cleaned_df, options['scaling_method'])

        if options['remove_low_variance']:
            cleaned_df = remove_low_variance(cleaned_df, options['variance_threshold'])

        if options['handle_skewness']:
            cleaned_df = handle_skewness(clean_dataframe, options['skew_threshold'])

        return cleaned_df

    except Exception as e:
        st.error(f"An error occurred during data cleaning: {str(e)}")
        return None

def remove_duplicates(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")
    
    return df.drop_duplicates()

def handle_missing(df, missing_numeric_method):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                if missing_numeric_method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif missing_numeric_method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                else:  # mode
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def handle_outliers(df, outlier_method):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if outlier_method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        elif outlier_method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col].mask(z_scores > 3, df[col].median())
    return df
    
def normalize_data(df, scaling_method):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if scaling_method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:  # standard
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = (df[col] - mean_val) / std_val
    return df
    
def remove_low_variance(df, variance_threshold):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")
    
    variance = df.var()
    df = df.loc[:, variance > variance_threshold]
    return df
    
def handle_skewness(df, skew_threshold):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data is not a pandas DataFrame. Type: {type(df)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if abs(stats.skew(df[col])) > skew_threshold:
            df[col] = np.log1p(df[col] - df[col].min() + 1)
    return df

def get_ai_cleaning_suggestions(data):
    client = get_openai_client()
    try:
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
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data modification assistant. Provide Python code to modify the data based on the user's request."},
            {"role": "user", "content": f"Given the following data:\n\n{data.to_string()}\n\nUser request: {prompt}\n\nProvide Python code to modify the data:"}
        ]
    )
    return response.choices[0].message.content

def generate_download_link(df, filename="data_report.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data Report</a>'
    return href

def show(project_name):
    st.header(f"Data Cleaning for Project: {project_name}")

    # Remove this line
    # chat_button = st.button("Chat with AI", key="chat_button")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    if 'data_sources' not in st.session_state.projects[project_name] or not st.session_state.projects[project_name]['data_sources']:
        st.warning("No data available. Please upload data first.")
        return

    # Initialize cleaned_data_sources if it doesn't exist
    if 'cleaned_data_sources' not in st.session_state.projects[project_name]:
        st.session_state.projects[project_name]['cleaned_data_sources'] = {}

    # Create tabs for each data source
    data_sources = st.session_state.projects[project_name]['data_sources']
    tabs = st.tabs(list(data_sources.keys()))

    for i, (source, data) in enumerate(data_sources.items()):
        with tabs[i]:
            st.subheader(f"Data from {source}")
            if not isinstance(data, pd.DataFrame):
                st.error(f"The data from {source} is not in the correct format. Please upload a valid dataset.")
                continue

            # Display original and editable data
            if source == 'CSV':
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Data")
                    st.dataframe(data, height=400)

                with col2:
                    st.subheader("Edit Data")
                    edited_data = st.data_editor(data, num_rows="dynamic", key=f"editor_{project_name}_{source}_original", height=400)
            else:
                st.subheader("Original Data")
                st.dataframe(data)

                st.subheader("Edit Data")
                edited_data = st.data_editor(data, num_rows="dynamic", key=f"editor_{project_name}_{source}_original")

            if not edited_data.equals(data):
                st.session_state.projects[project_name]['data_sources'][source] = edited_data
                st.success(f"Data from {source} updated successfully!")

            data = edited_data  # Use the edited data for further processing

            options = {}  # Initialize options dictionary

            with st.expander("Data Cleaning Options", expanded=False):
                st.subheader("Data Cleaning Options")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### Basic Cleaning")
                    options['remove_duplicates'] = st.checkbox("Remove Duplicate Rows", key=f"{source}_remove_duplicates")
                    options['handle_missing'] = st.checkbox("Handle Missing Values", key=f"{source}_handle_missing")
                    if options['handle_missing']:
                        options['missing_numeric_method'] = st.selectbox("Missing Numeric Values Method", ['mean', 'median', 'mode'], key=f"{source}_missing_numeric_method")
                        options['missing_categorical_method'] = st.selectbox("Missing Categorical Values Method", ['mode', 'constant', 'ffill', 'bfill'], key=f"{source}_missing_categorical_method")
                        if options['missing_categorical_method'] == 'constant':
                            options['missing_fill_value'] = st.text_input("Fill Value for Categorical", key=f"{source}_missing_fill_value")
                    options['remove_low_variance'] = st.checkbox("Remove Low Variance Features", key=f"{source}_remove_low_variance")
                    if options['remove_low_variance']:
                        options['variance_threshold'] = st.slider("Variance Threshold", 0.0, 1.0, 0.1, 0.01, key=f"{source}_variance_threshold")

                with col2:
                    st.markdown("### Outlier Handling")
                    options['handle_outliers'] = st.checkbox("Handle Outliers", key=f"{source}_handle_outliers")
                    if options['handle_outliers']:
                        options['outlier_method'] = st.selectbox("Outlier Handling Method", ['IQR', 'zscore', 'winsorize'], key=f"{source}_outlier_method")
                        if options['outlier_method'] == 'IQR':
                            options['iqr_multiplier'] = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key=f"{source}_iqr_multiplier")
                        elif options['outlier_method'] == 'zscore':
                            options['zscore_threshold'] = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1, key=f"{source}_zscore_threshold")
                        elif options['outlier_method'] == 'winsorize':
                            options['winsorize_limits'] = st.slider("Winsorize Limits", 0.0, 0.5, (0.05, 0.95), 0.01, key=f"{source}_winsorize_limits")

                    st.markdown("### Skewness Handling")
                    options['handle_skewness'] = st.checkbox("Handle Skewness", key=f"{source}_handle_skewness")
                    if options['handle_skewness']:
                        options['skew_threshold'] = st.slider("Skewness Threshold", 0.0, 2.0, 0.5, 0.1, key=f"{source}_skew_threshold")
                        options['skew_method'] = st.selectbox("Skewness Handling Method", ['log', 'sqrt', 'box-cox'], key=f"{source}_skew_method")

                with col3:
                    st.markdown("### Data Transformation")
                    options['normalize_data'] = st.checkbox("Normalize/Scale Data", key=f"{source}_normalize_data")
                    if options['normalize_data']:
                        options['scaling_method'] = st.selectbox("Scaling Method", ['minmax', 'standard', 'robust', 'quantile'], key=f"{source}_scaling_method")
                    
                    options['encode_categorical'] = st.checkbox("Encode Categorical Variables", key=f"{source}_encode_categorical")
                    if options['encode_categorical']:
                        options['encoding_method'] = st.selectbox("Encoding Method", ['one-hot', 'label', 'ordinal'], key=f"{source}_encoding_method")
                    
                    options['feature_selection'] = st.checkbox("Perform Feature Selection", key=f"{source}_feature_selection")
                    if options['feature_selection']:
                        options['feature_selection_method'] = st.selectbox("Feature Selection Method", ['correlation', 'mutual_info', 'chi2'], key=f"{source}_feature_selection_method")
                        options['feature_selection_threshold'] = st.slider("Feature Selection Threshold", 0.0, 1.0, 0.5, 0.01, key=f"{source}_feature_selection_threshold")

                if st.button("Apply Data Cleaning", key=f"{source}_apply_cleaning"):
                    cleaned_data = clean_dataframe(data, options)
                    if cleaned_data is not None:
                        st.session_state.projects[project_name]['cleaned_data_sources'][source] = cleaned_data
                        st.success(f"Data cleaning for {source} applied and saved successfully!")

                        st.subheader("Cleaned Data")
                        st.dataframe(cleaned_data)

                        # Create two columns for summary and download button
                        col1, col2 = st.columns([2, 1])

                        with col1:
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

                        with col2:
                            csv = cleaned_data.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'data:file/csv;base64,{b64}'
                            st.download_button(
                                label="Download Data Report",
                                data=csv,
                                file_name=f"{source}_cleaned_data_report.csv",
                                mime="text/csv",
                                key=f"{source}_download_report"
                            )

                    else:
                        st.error(f"Data cleaning for {source} failed. Please check your data and try again.")

            # AI Suggestions (outside the expander)
            st.subheader("AI Cleaning Suggestions")
            if st.button("Get AI Cleaning Suggestions", key=f"{source}_ai_suggestions"):
                with st.spinner("Getting AI suggestions..."):
                    suggestions = get_ai_cleaning_suggestions(data)
                    st.write(suggestions)

            # Modify Data with AI (outside the expander)
            st.subheader("Modify Data with AI")
            modification_prompt = st.text_input("Describe how you want to modify the data:", key=f"{source}_modification_prompt")
            if modification_prompt:
                modification_code = modify_data_with_ai(modification_prompt, data)
                st.code(modification_code, language="python")
                if st.button("Apply Modification", key=f"{source}_apply_modification"):
                    try:
                        exec(modification_code)
                        st.session_state.projects[project_name]['data_sources'][source] = data
                        st.success(f"Data for {source} modified successfully!")
                        st.dataframe(data)
                    except Exception as e:
                        st.error(f"Error modifying data for {source}: {str(e)}")

    # Chat functionality
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize chat_open if it doesn't exist
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = False

    # Floating chat button (with a unique key)
    chat_button = st.button("💬 Chat with Your Data", key="floating_chat_button")

    if chat_button:
        st.session_state.chat_open = not st.session_state.chat_open

    # Floating chat interface
    if st.session_state.chat_open:
        chat_container = st.container()
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Type your message here..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = get_ai_response(prompt, project_name)
                    message_placeholder.markdown(full_response)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def get_ai_response(prompt, project_name):
    client = get_openai_client()
    try:
        # Prepare context from raw data
        context = "Available data sources:\n"
        for source, data in st.session_state.projects[project_name]['data_sources'].items():
            context += f"- {source}: {data.shape[0]} rows, {data.shape[1]} columns\n"
            context += f"Columns: {', '.join(data.columns)}\n"
            context += f"Sample data:\n{data.head().to_string()}\n\n"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
{
    "role": "system",
    "content": "You are an expert data scientist and supply chain analyst with specialized knowledge in advanced data cleaning, statistical analysis, predictive modeling, and optimization techniques specifically for supply chain operations. Your responses should provide actionable, detailed, and specific recommendations to improve supply chain efficiency, reliability, and decision-making. Ensure all advice is supported by statistical reasoning and directly applicable to the provided data context."
},
{
    "role": "user",
    "content": f"Context:\n{context}\n\nUser question: {prompt}\n\nPlease provide detailed and specific guidance, including step-by-step recommendations and justifications based on the data context."
}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    show(st.session_state.current_project)
