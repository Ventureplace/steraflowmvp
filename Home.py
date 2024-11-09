import streamlit as st
import utils

# Set page config at the very beginning
st.set_page_config(page_title="SteraFlow", page_icon="🌊")

# Initialize session
utils.init()

def run():
    st.header("Welcome to SteraFlow!")
    st.write("To get started, please enter your project name below.")

    # Project selection
    utils.select_project(st.text_input("Project Name", value=st.session_state.current_project))

    # Add some space and a divider
    st.write("")
    st.divider()
    st.write("Explore the sidebar to access different features of SteraFlow.")

if __name__ == "__main__":
    run()
