import streamlit as st
from prep import initialize_openai_client, clean_dataframe, show as show_prep
import dashboard
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Default Project"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize OpenAI client
client = initialize_openai_client()

st.title("For Demo Only")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Clean Data", "Dashboard"])

# Project selection
project_name = st.sidebar.text_input("Project Name", value=st.session_state.current_project)
st.session_state.current_project = project_name

if project_name not in st.session_state.projects:
    st.session_state.projects[project_name] = {'data': None, 'cleaned_data': None}

def get_data_insights(data):
    global client
    if client is None:
        st.error("OpenAI client is not initialized. Please enter your API key.")
        return "Unable to get data insights at this time."
    
    data_description = data.describe().to_string()
    prompt = f"Given the following dataset description, provide some insights and suggestions for data cleaning:\n\n{data_description}\n\nInsights and suggestions:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

if page == "Upload Data":
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.projects[project_name]['data'] = data
        st.write("Data uploaded successfully!")
        
        st.subheader("Uploaded Data")
        st.dataframe(data)  # Display the uploaded data as a dataframe
        
        st.subheader("Edit Data")
        edited_data = st.data_editor(data, num_rows="dynamic")
        if not edited_data.equals(data):
            st.session_state.projects[project_name]['data'] = edited_data
            st.success("Data updated successfully!")
            
            st.subheader("Updated Data")
            st.dataframe(edited_data)  # Display the updated data as a dataframe

elif page == "Clean Data":
    show_prep(project_name)  # This now includes both dataframe and data editor functionality

elif page == "Dashboard":
    dashboard.show(project_name)

# Check OpenAI client status
if client is not None:
    st.sidebar.success("OpenAI client initialized")
else:
    st.sidebar.warning("OpenAI client not initialized. Some features may be limited.")
