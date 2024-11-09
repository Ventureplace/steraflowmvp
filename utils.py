import streamlit as st
from openai import OpenAI
from streamlit_extras.app_logo import add_logo

DATA_SOURCES = {'CSV': {'icon': None}, 
                'Public Google Sheets': {'icon': None},
                'Looker': {'icon': None},
                'Private Google Sheets': {'icon': None},
                'AWS S3': {'icon': None},
                'Firestore': {'icon': None},
                'Google Cloud Storage': {'icon': None},
                'Microsoft SQL Server': {'icon': None},
                'MongoDB': {'icon': None},
                'MySQL': {'icon': None},
                'PostgreSQL': {'icon': None},
                'Supabase': {'icon': None},
                # 'Tableau': {'icon': None},
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
    if 'chat_assistant' not in st.session_state:
        st.session_state.chat_assistant = None

def update_sidebar():
    add_logo("./assets/logo.png", height=150)

    # Display current project on sidebar
    st.sidebar.header("Current Session Info")
    curr_proj = st.session_state.current_project if 'current_project' in st.session_state else 'None'
    st.sidebar.write(f"Project: {curr_proj}")

def select_project(project_name):
    if project_name:
        st.session_state.current_project = project_name
        if project_name not in st.session_state.projects:
            st.session_state.projects[project_name] = {"data_sources": {}}

def init():
    init_session_state()
    update_sidebar()

