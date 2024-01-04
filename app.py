from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationRetrievalChain




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n ", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    conversation = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationRetrievalChain.from_llm(llm=llm, retriever=vector_store.as.retriever, memory=memory)

    return conversation_chain




def main():
    load_dotenv()
    #st.set_page_config(page_title="Find Your Candidate", page_icon="🇺🇸", layout="wide")
    st.set_page_config(page_title="Find Your Candidate", page_icon="Iraq", layout="wide")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Find Your Candidate")
    st.text_input("Ask a question about a candidate:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs of your resume/CVs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            st.spinner("Processing...")
            #get pdf text
            raw_text = get_pdf_text(pdf_docs)
            

            #get text chunks
            text_chunks = get_text_chunks(raw_text)
            
            #create vector store   
            vector_store = get_vector_store(text_chunks)


            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vector_store)
    
    

    

if __name__ == "__main__":
    main()  # execute only if run as a script