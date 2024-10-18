import streamlit as st
import utils
from openai import OpenAI

# Initialize session
utils.init()

def get_openai_client():
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return OpenAI(api_key=st.session_state.openai_api_key)
    return None

def run():
    st.header("Welcome to SteraFlow!")
    st.write("To start, please select a project and enter your OpenAI API key.")

    # Project selection
    utils.select_project(st.text_input("Project Name", value=st.session_state.current_project))

    # OpenAI API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password", key="api_key_input")
    
    if api_key:
        st.session_state["openai_api_key"] = api_key
        st.success("API key set successfully!")

    # Initialize OpenAI client
    client = get_openai_client()
    if client:
        st.session_state["client"] = client
        st.success("OpenAI client initialized successfully!")
    else:
        st.warning("Please enter a valid OpenAI API key to initialize the client.")

if __name__ == "__main__":
    run()
