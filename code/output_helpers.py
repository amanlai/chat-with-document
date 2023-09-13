import streamlit as st
import question_and_answer as QA


def collect_history(last_qa):

    # prepend the latest Q/A to the Q/A history
    sep = '-'*100 + '\n'
    st.session_state['history'] = last_qa + sep + st.session_state.get('history', '')

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

        # text area widget for the question and answer
        st.text_area('Answer:', value=answer)
        history = st.checkbox("Show history")

        st.divider()

        # the current question and answer
        last_qa = f'Q: {question} \nA: {answer} \n'

        # only show history if history=True
        if history:
            collect_history(last_qa)