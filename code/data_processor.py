import os
import tiktoken
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


class ProcessData:

    def __init__(self, file, chunk_size=256, chunk_overlap=10):
        
        load_dotenv(find_dotenv(), override=True)
        self.file = file
        self.data = self.load_document()
        self.chunks = self.chunk_data(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_cost = self.compute_embedding_cost()
        self.vector_store = self.build_embeddings()


    def load_document(self):
        """
        Load PDF, DOCX or TXT files as LangChain Documents
        """
        name, extension = os.path.splitext(self.file)
        if extension in ('.pdf', '.docx', '.txt'):
            # print(f'Loading {self.file}')
            if extension == '.pdf':
                loader = PyPDFLoader(self.file)
            elif extension == '.docx':
                loader = Docx2txtLoader(self.file)
            else:
                loader = TextLoader(self.file, encoding='utf-8')
        else:
            raise ValueError('Document format is not supported!')

        data = loader.load()
        return data


    def chunk_data(self, chunk_size, chunk_overlap):
        """
        Chunk the data and return chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.data)
        return chunks


    def compute_embedding_cost(self, model='text-embedding-ada-002'):
        """
        Compute the total number of tokens and its embedding cost
        """
        enc = tiktoken.encoding_for_model(model)
        total_tokens = sum(len(enc.encode(page.page_content)) for page in self.chunks)
        return {
            'Total Tokens': total_tokens, 
            'Embedding Cost in USD': total_tokens / 1000 * 0.0004}



    def build_embeddings(self):
        """
        Create embeddings and save them in a Chroma vector store
        """
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(self.chunks, embeddings)
        return vector_store
