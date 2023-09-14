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


def display_answer(question, answer):

    # text area widget for the question and answer
    st.text_area('Answer:', value=answer)
    history = st.checkbox("Show history")

    st.divider()

    # only show history if history=True
    if history:
        # the current question and answer
        last_qa = f'Q: {question} \nA: {answer} \n'
        collect_history(last_qa)



def ask_question(ques, k, history=True):
    # continue only if there's a vector store
    vs = st.session_state.get('vector_store', None)
    if vs is not None:
        ans = QA.get_answer(vs, ques, k=k)
        display_answer(ques, ans)



def ask_question_with_memory(ques, k):
    # continue only if there's a vector store
    vs = st.session_state.get('vector_store', None)
    if vs is not None:
        ch = st.session_state.get('chat_history', [])
        output = QA.get_answer_with_memory(vs, ques, chat_history=ch, k=k)
        st.session_state.setdefault('chat_history', []).append((ques, output['answer']))
        display_answer(ques, output['answer'])