import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    """Splits text into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store using Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    """Creates a conversational retrieval chain with memory using Gemini 1.5 Flash."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vector_store.as_retriever()
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
