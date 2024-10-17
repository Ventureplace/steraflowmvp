import streamlit as st
import utils

# Initialize session
utils.init()

def run():
    st.header("Welcome to SteraFlow!")
    st.write("To start, please select a project and enter your OpenAI API key.")

    # Project selection
    utils.select_project(st.text_input("Project Name", value=st.session_state.current_project))

    # Initialize OpenAI client
    utils.initialize_openai_client()


if __name__ == "__main__":
    run()