import streamlit as st
import utils
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import os

# Initialize session
utils.init()

def on_src_change(src):
    st.session_state.src = src

def get_csv_data():
    data = None
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Save the file name in session state
        if 'recent_files' not in st.session_state:
            st.session_state.recent_files = []
        st.session_state.recent_files.append(uploaded_file.name)
        st.session_state.recent_files = st.session_state.recent_files[-5:]  # Keep only the last 5 files
    return data

def get_pub_sheets_data():
    data = None
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
    return data

def show(project_name):
    st.header(f"Upload Data for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    # Add a grid of buttons for each data source
    st.write("Select a Source")
    NUM_COLS = 3
    curr_col = 0
    cols = st.columns(NUM_COLS)
    for src in utils.DATA_SOURCES:
        cols[curr_col].button(src, key=src, on_click=on_src_change, args=[src], icon=utils.DATA_SOURCES[src]['icon'], use_container_width=True)
        curr_col = (curr_col + 1) % NUM_COLS
    st.divider()

    # Get data from user specified source
    new_data = None
    if st.session_state.src == 'CSV':
        new_data = get_csv_data()
    elif st.session_state.src == 'Public Google Sheets':
        new_data = get_pub_sheets_data()
    else:
        st.write(f"Source not yet supported: {st.session_state.src}")

    # Store and display the retrieved data
    if new_data is not None:
        st.session_state.projects[project_name]['data'] = new_data
        st.write("Data uploaded successfully!")

    # Display current data and recent files side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Data")
        if st.session_state.projects[project_name]['data'] is not None:
            st.dataframe(st.session_state.projects[project_name]['data'], height=400)
        else:
            st.write("No data uploaded yet.")

    with col2:
        st.subheader("Recent Files")
        if 'recent_files' in st.session_state and st.session_state.recent_files:
            selected_file = st.selectbox("Select a recent file", st.session_state.recent_files)
            if st.button("Load Selected File"):
                try:
                    loaded_data = pd.read_csv(selected_file)
                    st.session_state.projects[project_name]['data'] = loaded_data
                    st.success(f"Loaded data from {selected_file}")
                    # Refresh the current data display
                    with col1:
                        st.subheader("Current Data")
                        st.dataframe(loaded_data, height=400)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        else:
            st.write("No recent files available.")

if __name__ == "__main__":
    show(st.session_state.current_project)
