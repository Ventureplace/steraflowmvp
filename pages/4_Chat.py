import streamlit as st
from chat_assistant import ChatAssistant
import utils

# Initialize session
utils.init()

def show():
    col1, col2 = st.columns([4, 1])
    col1.header(f"Chat")
    col2.button('Reset Chat', on_click=reset_conversation)

    # Restore chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user prompt and send back response
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = st.session_state.chat_assistant.get_response(prompt)
            response = st.write_stream(stream)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

def reset_conversation():
  st.session_state.chat_history = []
  st.session_state.chat_assistant.renew_thread()

if __name__ == "__main__":
    if st.session_state.chat_assistant is None:
        st.session_state.chat_assistant = ChatAssistant(st.secrets['OPENAI_API_KEY'])
    show()