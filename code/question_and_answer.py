from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


def get_answer(vector_store, question, model='gpt-3.5-turbo', k=3):
    """
    Given vector store use cosine similarity via LangChain
    to return an answer
    """
    llm = ChatOpenAI(model=model, temperature=0.1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(question)
    return answer


def ask_with_memory(vector_store, question, model='gpt-3.5-turbo', chat_history=[], k=5):

    llm = ChatOpenAI(model=model, temperature=0.1)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result, chat_history