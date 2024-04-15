import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from cromadbTest import load_data , execute_query ,load_pdf_data
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
import hmac
import requests

# from langchain.llms import HuggingFaceHub
load_dotenv()
st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")

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


def handle_userinput(user_question):
    # Ensure 'chat_history' is initialized as a list if it does not exist or is None
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Execute the query to get the response
    response = execute_query(user_question)

    # Append the question and response to the session_state chat_history
    st.session_state.chat_history.append({"question": user_question, "response": response})
    output_list = []
    for message in st.session_state.chat_history:
        output_list.append({"question": message["question"], "response": message["response"]})
    output_list.reverse()
    # Display the chat history
    for message in output_list:
        st.write(user_template.replace("{{MSG}}", message["question"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message["response"]), unsafe_allow_html=True)


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

def registration_page():
    st.title('Registration Page')

    with st.form("registration_form"):
        app_name = st.text_input("App Name")
        password = st.text_input("Password", type="password")
        
        # Form submission button
        submitted = st.form_submit_button("Register")
        if submitted:
            # Here, you would typically add your code to register the user
            # For example, saving the user data to a database
            # Now also store the password using the new function
            if store_password(app_name, password):
                st.success(f"Account created for {app_name}!")
            else:
                st.error("Failed to create account. Please try again.")

def main():
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Best Candidate AI")
    
    
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
                            # Comment out or remove the line that displays the DataFrame
                            # st.write(df)
                    # Add a message indicating the files have been processed
                    st.success("Files processed successfully.")

def store_password(app_name, password):
    # Assuming you have an endpoint to store the password
    response = requests.post(f"http://vaibhavsharma3070.pythonanywhere.com/store_password", data={'app_name': app_name, 'password': password})
    if response.status_code == 200:
        return True  # Or any other success criteria
    else:
        st.error("Failed to store password")
        return False

def get_password(application_id):
    response = requests.get(f"http://vaibhavsharma3070.pythonanywhere.com/get_password?app_name={application_id}")
    password = response.text.strip()
    if password == "Password not found":
        st.error("Invalid application_id")
    return password

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        password_from_api = get_password(st.session_state["application_id"])
        print("password_from_api == ",password_from_api)
        if hmac.compare_digest(st.session_state["password"], password_from_api):
            st.session_state["password_correct"] = True
            st.session_state["logged_in"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["logged_in"] = False
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True
    st.markdown("<h1 style='text-align: center; color: black;'>Best Candidate AI</h1>", unsafe_allow_html=True)   # Show input for password.
    st.text_input("application_id", type='default', key='application_id')
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return False


if __name__ == '__main__':
    # Check if the user is already logged in
    if not st.session_state.get('logged_in', False):
        page = st.sidebar.selectbox("Choose a page", ["Login", "Registration"])
    else:
        page = "Main"  # Directly go to the main page if already logged in

    if page == "Login":
        if not check_password():
            st.stop()  # Do not continue if check_password is not True.
        else:
            # Set the logged_in state to True upon successful login
            st.session_state.logged_in = True
            main()
    elif page == "Registration":
        registration_page()
    elif page == "Main":
        main()
