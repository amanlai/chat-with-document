import os
import streamlit as st
from data_processor import ProcessData, load_dotenv, find_dotenv
import question_and_answer as QA


# clear the Q/A history from streamlit session state
def clear_history():
    st.session_state.pop('history', None)


# clear the input field upon hitting Enter
def clear_input():
    st.session_state['latest_question'] = st.session_state['input_field']
    st.session_state['input_field'] = ''
    



def get_user_input():
    """
    Get file, chunk_size and k as user input on the sidebar
    """
    # file uploader widget
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
    # chunk size input widget
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
    # k input widget
    k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
    # add data button widget
    add_data = st.button('Add Data', on_click=clear_history)
    return chunk_size, k, uploaded_file, add_data


def build_data(uploaded_file, chunk_size):
    """
    Process data on the sidebar
    """
    with st.spinner('Reading, splitting and embedding file...'):

        # writing the file from RAM to the current directory on disk
        bytes_data = uploaded_file.read()
        file_name = os.path.join('../data/sample_data/', uploaded_file.name)
        with open(file_name, 'wb') as f:
            f.write(bytes_data)

        process_data = ProcessData(file_name, chunk_size=chunk_size)
        st.write(f'Chunk size: {chunk_size}, Chunks: {len(process_data.chunks)}')

        cost = process_data.embedding_cost
        st.write(f"Embedding Cost: ${cost['Embedding Cost in USD']:.4f}")

        # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state['vector_store'] = process_data.vector_store
        st.success('File uploaded, chunked and embedded successfully.')


def collect_history(last_qa):

    # prepend the latest Q/A to the Q/A history
    st.session_state['history'] = f"{last_qa} \n {'-' * 100} \n {st.session_state.get('history', '')}"

    # text area widget for the Q/A history
    st.text_area(
        label='Q/A History', 
        value=st.session_state['history'], 
        key='history', 
        height=400
        )


def ask_question(question, k, history=True):
    # continue only if there's a vector store
    if 'vector_store' in st.session_state:
        vector_store = st.session_state['vector_store']
        answer = QA.get_answer(vector_store, question, k=k)

        # text area widget for the answer
        st.text_area('Answer:', value=answer)

        st.divider()

        # the current question and answer
        last_qa = f'Q: {question} \nA: {answer}'

        # only show history if history=True
        if history:
            collect_history(last_qa)



def create_sidebar():

    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        chunk_size, k, uploaded_file, add_data = get_user_input()

        if uploaded_file and add_data: # if the user browsed a file
            build_data(uploaded_file, chunk_size)

    return k

    

if __name__ == '__main__':

    # loading the OpenAI api key from .env
    load_dotenv(find_dotenv(), override=True)

    st.image('../data/img.png')
    st.subheader('Q/A on Private Documents')
    k = create_sidebar()

    # user's question input widget
    # the input field clears after hitting Enter
    st.text_input(
        'Ask a question based on the content of the file you uploaded:', 
        key='input_field',
        on_change=clear_input)
    question = st.session_state.get('latest_question', '')

    # proceed only if the user entered a question
    if question:
        ask_question(question, k)

# run the app: streamlit run main.py