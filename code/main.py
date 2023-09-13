import os
import streamlit as st
from data_processor import load_dotenv, find_dotenv

from input_helpers import (
    create_sidebar, 
    get_user_input, 
#    clear_input, 
    build_data
    )
from output_helpers import ask_question


    






    

if __name__ == '__main__':

    # loading the OpenAI api key from .env
    load_dotenv(find_dotenv(), override=True)

    st.image('../data/img.png')
    st.subheader('Q/A on Private Documents')
    k = create_sidebar()

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
        ask_question(question, k)

# run the app: streamlit run main.py