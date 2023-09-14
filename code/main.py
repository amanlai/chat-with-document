import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from input_helpers import create_sidebar
from output_helpers import ask_question, ask_question_with_memory


    
if __name__ == '__main__':

    # loading the OpenAI api key from .env
    load_dotenv(find_dotenv(), override=True)

    st.image('../data/img.png')
    st.subheader('Q/A on Private Documents')
    k, with_memory = create_sidebar()

    # user's question input widget
    # the input field clears after hitting Enter
    # st.text_input(
    #     'Ask a question based on the content of the file you uploaded:', 
    #     key='input_field')
    # question = st.session_state.get('latest_question', '')
    question = st.text_input(
        'Ask a question based on the content of the file you uploaded:'
        )

    # proceed only if the user entered a question
    if question:
        if with_memory:
            ask_question_with_memory(question, k)
        else:
            ask_question(question, k)

# run the app: streamlit run main.py