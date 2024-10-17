import streamlit as st
from prep import initialize_openai_client, clean_dataframe, show as show_prep
import dashboard
import pandas as pd
import plotly.express as px
from openai import OpenAI
from streamlit_gsheets import GSheetsConnection

data_sources = {'CSV': {'icon': ':material/mood:'}, 
                'Public Google Sheets': {'icon': ':material/mood:'},
                # 'Private Google Sheets': {'icon': ':material/mood:'},
                'AWS S3': {'icon': ':material/mood:'},
                'Firestore': {'icon': ':material/mood:'},
                'Google Cloud Storage': {'icon': ':material/mood:'},
                # 'Microsoft SQL Server': {'icon': ':material/mood:'},
                # 'MongoDB': {'icon': ':material/mood:'},
                # 'MySQL': {'icon': ':material/mood:'},
                # 'Neon': {'icon': ':material/mood:'},
                # 'PostgreSQL': {'icon': ':material/mood:'},
                # 'Snowflake': {'icon': ':material/mood:'},
                # 'Supabase': {'icon': ':material/mood:'},
                # 'Tableau': {'icon': ':material/mood:'},
                # 'TiDB': {'icon': ':material/mood:'},
                'TigerGraph': {'icon': ':material/mood:'}}

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = "Default Project"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'src' not in st.session_state:
    st.session_state.src = next(iter(data_sources))

def on_src_change(src):
    st.session_state.src = src

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

if page == "Upload Data":
    st.header("Upload Data")
    
    # Add a grid of buttons for each data source
    st.write("Select a Source")
    NUM_COLS = 3
    curr_col = 0
    cols = st.columns(NUM_COLS)
    for src in data_sources:
        data_sources[src]['btn'] = cols[curr_col].button(src, key=src, on_click=on_src_change, args=[src], icon=data_sources[src]['icon'], use_container_width=True)
        curr_col = (curr_col + 1) % NUM_COLS
    st.divider()

    # Get data from user specified source
    data = None
    if st.session_state.src == 'CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    elif st.session_state.src == 'Public Google Sheets':
        st.write("Create a Google Sheets share link with the data you would like to upload. The link should have \"Anyone with the link\" set as a \"Viewer.\"")
        sheet_url = st.text_input("Sheet URL*")
        sheet_name = st.text_input("Sheet Name*")
        if st.button('Upload'):
            if sheet_url and sheet_name:
                try:
                    # Create a connection object and read spreadsheet
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    data = conn.read(spreadsheet=sheet_url) #, worksheet=sheet_name)    # fix worksheet specification
                except Exception as e:
                    st.error(f"Error reading sheet: {str(e)}")
            else:
                st.write('Please fill all requred fields (*)')
    else:
        st.write(f"Source not yet supported: {st.session_state.src}")

    # Store and display the retrieved data
    if data is not None:
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
