import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from cromadbTest import cromadb_test ,load_data , execute_query ,load_pdf_data
import pandas as pd
import csv
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Modify the handle_userinput function to store both questions and responses
chat_history = []
def handle_userinput(user_question):
    response = execute_query(user_question)    
    chat_history.append({"question": user_question, "response": response})
    
    for i, message in enumerate(chat_history):        
        st.write(user_template.replace("{{MSG}}", message["question"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message["response"]), unsafe_allow_html=True)

# chat_history = []
# def handle_userinput(user_question):
        
#     response = execute_query(user_question)
#     # st.session_state.chat_history = response
#     chat_history.append(response)
    
#     for i, message in enumerate(chat_history):
#         print("Message detail : ",message)
#         print("Message detail i : ",i)
#         st.write(user_template.replace(
#                 "{{MSG}}", message), unsafe_allow_html=True)
        # if i % 2 == 0:
        #     st.write(user_template.replace(
        #         "{{MSG}}", message), unsafe_allow_html=True)
        # else:
        #     st.write(bot_template.replace(
        #         "{{MSG}}", message), unsafe_allow_html=True)


def csv_to_text(csv_file):
    text_data = ""
    with open(csv_file, 'r',encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            text_data += ' '.join(row) + '\n'
    return text_data

def read_csv_file(csv_file):
    if csv_file is not None:
        # To read csv files as a pandas DataFrame
        return pd.read_csv(csv_file)
    return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Find Your Candidate")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template.replace("{{MSG}}","Hello BestCandidate AI"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human from SES."), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        csv_docs = st.file_uploader("Upload database CSV file and click on 'Process'", type=["csv"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Add validation for file uploads before processing
                if pdf_docs is None or csv_docs is None:
                    st.error("Please upload both PDF and CSV files before processing.")
                else:
                    # Continue with processing the files
                    
                    if pdf_docs:
                        raw_text = get_pdf_text(pdf_docs) 
                        # get the text chunks
                        load_pdf_data(raw_text)
                        text_chunks = get_text_chunks(raw_text)

                        # # create vector store
                        # vectorstore = get_vectorstore(text_chunks)
                        # print("vector storage:",vectorstore)
                        # # create conversation chain
                        # st.session_state.conversation = get_conversation_chain(
                        #     vectorstore)
                # Process CSV files
                    if csv_docs:                
                        for csv_file in csv_docs:
                                                        
                            df = read_csv_file(csv_file)
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            file_name = f'csvdata/output_{timestamp}.csv'
                            df.to_csv(file_name, index=False)
                            load_data(file_name)                
                            raw_text_csv = csv_to_text(file_name)
                            text_chunks_csv = get_text_chunks(raw_text_csv)

                            # create vector store
                            vectorstore_csv = get_vectorstore(text_chunks_csv)

                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore_csv)
                            # df = read_csv_file(csv_file)
                            # Now you can process the DataFrame 'df' as needed
                            # For example, you can display the DataFrame in the app
                            st.write(df)

if __name__ == '__main__':
    main()
