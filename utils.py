import streamlit as st
from openai import OpenAI
from streamlit_extras.app_logo import add_logo

DATA_SOURCES = {'CSV': {'icon': None}, 
                'Public Google Sheets': {'icon': None},
                'Private Google Sheets': {'icon': None},
                'AWS S3': {'icon': None},
                'Firestore': {'icon': None},
                'Google Cloud Storage': {'icon': None},
                'Microsoft SQL Server': {'icon': None},
                'MongoDB': {'icon': None},
                'MySQL': {'icon': None},
                'PostgreSQL': {'icon': None},
                'Supabase': {'icon': None},
                'Tableau': {'icon': None},
                # 'Neon': {'icon': None},
                # 'Snowflake': {'icon': None},
                # 'TiDB': {'icon': None},
                # 'TigerGraph': {'icon': None},
                }

def init_session_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'current_project' not in st.session_state:
        select_project("Default Project")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'src' not in st.session_state:
        st.session_state.src = next(iter(DATA_SOURCES))
    if 'client' not in st.session_state:
        st.session_state.client = None

def initialize_openai_client():
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)
        st.success("API key entered successfully!")
    else:
        client = None
        st.error("Please enter a valid OpenAI API key to use the AI-powered features.")
    st.session_state.client = client
    

def get_data_insights(data):
    if 'client' in st.session_state and st.session_state.client is not None:
        client = st.session_state.client
    else:
        st.error("OpenAI client is not initialized. Please enter your API key.")
        return "Unable to get data insights at this time."
    
    data_description = data.describe().to_string()
    prompt = f"Given the following dataset description, provide some insights and suggestions for data cleaning:\n\n{data_description}\n\nInsights and suggestions:"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def update_sidebar():
    add_logo("./assets/logo.png", height=150)

    # Display current project on sidebar
    st.sidebar.header("Current Session Info")
    curr_proj = st.session_state.current_project if 'current_project' in st.session_state else 'None'
    st.sidebar.write(f"Project: {curr_proj}")

    # Check OpenAI client status
    if 'client' in st.session_state and st.session_state.client is not None:
        st.sidebar.success("OpenAI client initialized")
    else:
        st.sidebar.warning("OpenAI client not initialized. Some features may be limited.")

def select_project(project_name):
    st.session_state.current_project = project_name
    if project_name not in st.session_state.projects:
        st.session_state.projects[project_name] = {'data': None, 'cleaned_data': None}

def init():
    init_session_state()
    update_sidebar()