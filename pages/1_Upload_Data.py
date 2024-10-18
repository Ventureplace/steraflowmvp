import streamlit as st
import utils
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# Initialize session
utils.init()

def on_src_change(src):
    st.session_state.src = src

def get_csv_data():
    data = None
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
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
    data = new_data if new_data is not None else st.session_state.projects[project_name]['data']
    if data is not None:
        st.session_state.projects[project_name]['data'] = data

        # Show appropriate headings
        if new_data is not None:
            st.write("Data uploaded successfully!")

        # Display current data and edit data side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Data")
            st.dataframe(data, height=400)

        with col2:
            st.subheader("Edit Data")
            edited_data = st.data_editor(data, num_rows="dynamic", height=400)

        if not edited_data.equals(data):
            st.session_state.projects[project_name]['data'] = edited_data
            st.success("Data updated successfully!")
            
            # Update the current data display
            with col1:
                st.subheader("Updated Current Data")
                st.dataframe(edited_data, height=400)

        # Display cleaned data separately
        st.subheader("Cleaned Data")
        cleaned_data = st.session_state.projects[project_name].get('cleaned_data', pd.DataFrame())
        st.dataframe(cleaned_data, height=400)

if __name__ == "__main__":
    show(st.session_state.current_project)
