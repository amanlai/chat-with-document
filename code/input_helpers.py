import os
import streamlit as st
from data_processor import ProcessData
from tempfile import NamedTemporaryFile



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
    k = st.slider("Top-k:", value=3, min_value=1, max_value=20, step=1, on_change=clear_history)
    # ask whether to save data
    # save_data = st.selectbox("Save data on disk:", ['Yes', 'No'], on_change=clear_history) == 'Yes'
    with_memory = st.selectbox("With memory:", ['Yes', 'No'], on_change=clear_history) == 'Yes'
    # add data button widget
    add_data = st.button('Add Data', on_click=clear_history)
    return chunk_size, k, uploaded_file, add_data, with_memory



def create_sidebar():

    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        chunk_size, k, uploaded_file, add_data, with_memory = get_user_input()

        if uploaded_file and add_data: # if the user browsed a file
            build_data(uploaded_file, chunk_size)

    return k, with_memory



def build_data(uploaded_file, chunk_size):
    """
    Process data on the sidebar
    """
    with st.spinner('Reading, splitting and embedding file...'):

        # writing the file from RAM to the current directory on disk
        # bytes_data = uploaded_file.read()
        # file_name = os.path.join('../data/sample_data/', uploaded_file.name)
        # with open(file_name, 'wb') as f:
        #     f.write(bytes_data)

        with NamedTemporaryFile(delete=False) as tmp:
            ext = os.path.splitext(uploaded_file.name)[1]
            tmp.write(uploaded_file.read())
            process_data = ProcessData(tmp.name, ext, chunk_size=chunk_size)
        os.remove(tmp.name)
        
        st.write(f'Chunk size: {chunk_size}, Chunks: {len(process_data.chunks)}')

        cost = process_data.embedding_cost
        st.write(f"Embedding Cost: ${cost['Embedding Cost in USD']:.4f}")

        # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state['vector_store'] = process_data.vector_store
        st.success('File uploaded, chunked and embedded successfully.')
