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
        # Save the file name and data in session state
        if 'csv_files' not in st.session_state:
            st.session_state.csv_files = []
        st.session_state.csv_files.append((uploaded_file.name, data))
        st.session_state.csv_files = st.session_state.csv_files[-5:]  # Keep only the last 5 files
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
                # Save the sheet name and data in session state
                if 'sheets_files' not in st.session_state:
                    st.session_state.sheets_files = []
                st.session_state.sheets_files.append((sheet_name, data))
                st.session_state.sheets_files = st.session_state.sheets_files[-5:]  # Keep only the last 5 files
            except Exception as e:
                st.error(f"Error reading sheet: {str(e)}")
        else:
            st.write('Please fill all required fields (*)')
    return data

def get_looker_data():
    return None

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
    elif st.session_state.src == 'Looker':
        new_data = get_looker_data()
    else:
        st.write(f"Source not yet supported: {st.session_state.src}")

    # Store and display the retrieved data
    if new_data is not None:
        if 'data_sources' not in st.session_state.projects[project_name]:
            st.session_state.projects[project_name]['data_sources'] = {}
        st.session_state.projects[project_name]['data_sources'][st.session_state.src] = new_data
        st.write(f"Data from {st.session_state.src} uploaded successfully!")

    # Display current data and recent files side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Data")
        if 'data_sources' in st.session_state.projects[project_name]:
            for source, data in st.session_state.projects[project_name]['data_sources'].items():
                st.write(f"Data from {source}:")
                st.dataframe(data, height=200)
        else:
            st.write("No data uploaded yet.")

    with col2:
        st.subheader("Recent Files")
        tab1, tab2 = st.tabs(["CSV Files", "Google Sheets"])
        
        with tab1:
            if 'csv_files' in st.session_state and st.session_state.csv_files:
                file_names = [file[0] for file in st.session_state.csv_files]
                selected_file = st.selectbox("Select a recent CSV file", file_names, key="csv_select")
                if st.button("Load Selected CSV File"):
                    for file_name, file_data in st.session_state.csv_files:
                        if file_name == selected_file:
                            st.session_state.projects[project_name]['data_sources']['CSV'] = file_data
                            st.success(f"Loaded CSV data from {selected_file}")
                            break
            else:
                st.write("No recent CSV files available.")
        
        with tab2:
            if 'sheets_files' in st.session_state and st.session_state.sheets_files:
                sheet_names = [sheet[0] for sheet in st.session_state.sheets_files]
                selected_sheet = st.selectbox("Select a recent Google Sheet", sheet_names, key="sheet_select")
                if st.button("Load Selected Google Sheet"):
                    for sheet_name, sheet_data in st.session_state.sheets_files:
                        if sheet_name == selected_sheet:
                            st.session_state.projects[project_name]['data_sources']['Public Google Sheets'] = sheet_data
                            st.success(f"Loaded Google Sheets data from {selected_sheet}")
                            break
            else:
                st.write("No recent Google Sheets available.")

if __name__ == "__main__":
    show(st.session_state.current_project)