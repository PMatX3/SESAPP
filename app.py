import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from cromadbTest import load_data, execute_query, load_pdf_data, get_chat_history, load_json_data
import pandas as pd
import csv
import time, json
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import hmac
import streamlit as st
import threading
import requests

# from langchain.llms import HuggingFaceHub
load_dotenv()
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
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

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    # Ensure 'chat_history' is initialized as a list if it does not exist or is None
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Execute the query to get the response
    if st.session_state['temp']:
        response = execute_query(user_question, st.session_state["user_application_id"], True)
    else:
        response = execute_query(user_question, st.session_state["user_application_id"])

    # Append the question and response to the session_state chat_history
    st.session_state.chat_history.append(
        {"question": user_question, "response": response}
    )
    output_list = []
    for message in st.session_state.chat_history:
        output_list.append(
            {"question": message["question"], "response": message["response"]}
        )
    output_list.reverse()
    # Display the chat history
    for message in output_list:
        st.write(
            user_template.replace("{{MSG}}", message["question"]),
            unsafe_allow_html=True,
        )
        st.write(
            bot_template.replace("{{MSG}}", message["response"]), unsafe_allow_html=True
        )


def csv_to_text(csv_file):
    text_data = ""
    with open(csv_file, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            text_data += " ".join(row) + "\n"
    return text_data


def read_csv_file(csv_file):
    if csv_file is not None:
        # To read csv files as a pandas DataFrame
        return pd.read_csv(csv_file)
    return None


# def registration_page():
#     st.title("Sign Up")

#     with st.form("registration_form"):
#         app_name = st.text_input("User email")
#         password = st.text_input("Password", type="password")

#         # Form submission button
#         submitted = st.form_submit_button("Register")
#         if submitted:
#             # Here, you would typically add your code to register the user
#             # For example, saving the user data to a database
#             # Now also store the password using the new function
#             if store_password(app_name, password):
#                 st.success(f"Account created for {app_name}!")
#             else:
#                 st.error("Failed to create account. Please try again.")


def main():
    st.write(css, unsafe_allow_html=True)

    # Initialize session state for conversation and chat history if not already present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "temp" not in st.session_state:
        st.session_state['temp'] = False  # Initialize 'temp' to False

    # Assuming user_id is available and uniquely identifies the user
    # You need to determine how to obtain this ID. This could be from the session state, a login system, etc.
    if 'user_application_id' in st.session_state:
        user_id = st.session_state["user_application_id"]
    else:
        user_id = 0
    print(st.session_state["user_application_id"])
    # Retrieve and display the chat history for the user

    st.header("Your Best Candidate AI")

    # Input for user's question
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_input("Type your question about your documents here:", value=st.session_state.user_input)
    if st.button('Ask'):
        user_question = user_input  # This line assigns the input text to user_question only when the button is clicked
        st.session_state.user_input = ""

        # Now, you can use user_question as before
        if user_question:
            handle_userinput(user_question)

    chat_history = get_chat_history(user_id)
    if chat_history:
        st.write("Your previous conversations:")
        for message in chat_history:
            # Assuming 'message' is a dictionary with 'message', 'response', and possibly other keys like 'timestamp'
            st.write(user_template.replace("{{MSG}}", message["message"]), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", message["response"]), unsafe_allow_html=True)
    else:
        st.write("No previous conversations found.")

    # Example placeholders for displaying a welcome message
    st.write(
        user_template.replace("{{MSG}}", "Hello BestCandidate AI"),
        unsafe_allow_html=True,
    )
    st.write(
        bot_template.replace("{{MSG}}", "Hello Human from SES."), unsafe_allow_html=True
    )

    # Sidebar for document uploads (PDFs and CSVs)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        csv_docs = st.file_uploader(
            "Upload database CSV file and click on 'Process'",
            type=["csv"],
            accept_multiple_files=True,
        )

        # use_recrutly_data = st.checkbox("Use Recrutly.io data")
        use_recrutly_data = st.checkbox("Use SES data", value=False, key="use_recrutly_data")
    
        # Process button
        if st.button("Process"):
            with st.spinner("Processing"):
                # Validation for file uploads
                pdfloaded = False
                csvloaded = False
                if pdf_docs is None or csv_docs is None or use_recrutly_data is None:
                    st.error("Please upload both PDF and CSV files before processing.")
                else:
                    # Continue with processing the files
                    
                    if pdf_docs:
                        raw_text = get_pdf_text(pdf_docs) 
                        # get the text chunks
                        load_pdf_data(raw_text)
                        text_chunks = get_text_chunks(raw_text)
                        pdfloaded = True
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
                            csvloaded = True
                            # Comment out or remove the line that displays the DataFrame
                            # st.write(df)

                    if use_recrutly_data:
                        st.session_state['temp'] = True
                        
                        # with open('candidates_data.json', 'r', encoding='utf-8') as file:
                        #     data = json.load(file)
                        # load_json_data(data)
                        print('json data loaded')
                    if pdfloaded or csvloaded:
                        st.session_state['temp'] = False
                        st.success("Files processed.")
                    elif use_recrutly_data:
                        st.success("SES data processed.")
                    else:
                        st.error("Choose any one option from above for proceed.")


def load_and_cache_json_data():
    def load_json():
        with open('candidates_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        load_json_data(data)

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_json)
    thread.start()

def store_password(app_name, password):
    # Assuming you have an endpoint to store the password
    response = requests.post(
        f"http://vaibhavsharma3070.pythonanywhere.com/store_password",
        data={"app_name": app_name, "password": password},
    )
    if response.status_code == 200:
        return True  # Or any other success criteria
    else:
        st.error("Failed to store password")
        return False


def get_password(application_id):
    try:
        response = requests.get(
            f"http://vaibhavsharma3070.pythonanywhere.com/get_password?app_name={application_id}"
        )
        if response.status_code == 200:
            login_attempt = response.json().get('login', 0)  # Using .get() to avoid KeyError
            password = response.json().get('password', "Password not found")
            if password == "Password not found":
                st.error("Invalid application_id")
            return password, login_attempt
        else:
            st.error("😕 Invalid Email!")
            return None, 0  # Return None for password and 0 for login_attempt in case of error
    except requests.exceptions.JSONDecodeError:
        st.error("😕 Invalid Email!")
        return None, 0  # Return None for password and 0 for login_attempt in case of error

def change_password_page():
    st.title("Change Password")

    with st.form("change_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        submitted = st.form_submit_button("Change Password")
        if submitted:
            if new_password and new_password == confirm_password:
                valid, message = validate_password(new_password)
                if valid:
                    if store_new_password(st.session_state["user_application_id"], new_password):
                        st.success("Password changed successfully.")
                        # Update session state to reflect the successful login and password change
                        st.session_state["logged_in"] = True
                        st.session_state["require_password_change"] = False
                        # Optionally, you might want to reset or clear other session state flags here
                        
                        # Force a rerun of the app to reflect the changes immediately
                        st.experimental_rerun()
                    else:
                        st.error("Failed to change password.")
                else:
                    st.error(message)
            else:
                st.error("Passwords do not match.")

def validate_password(password):
    import re
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r'[0-9]', password):
        return False, "Password must include at least one number."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must include at least one special character."
    return True, ""

def store_new_password(application_id, new_password):
    response = requests.post(
        f"https://vaibhavsharma3070.pythonanywhere.com/update_password",
        data={"app_name": application_id, "new_password": new_password},
    )
    return response.status_code == 200

def update_login_attempts(application_id):
    """
    Updates the login attempt count for a given application ID.

    :param application_id: The application ID for which to update the login attempts.
    """
    response = requests.post(
        "https://vaibhavsharma3070.pythonanywhere.com/update_login_attempts",
        data={"app_name": application_id}
    )
    if response.status_code == 200:
        print("Login attempts updated successfully.")
    else:
        print("Failed to update login attempts.")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        password_from_api, login_attempt = get_password(st.session_state.get("application_id", ""))
        # Ensure password_from_api is not None before comparison
        if password_from_api is not None:
            if hmac.compare_digest(st.session_state["password"], password_from_api):
                st.session_state["password_correct"] = True
                st.session_state["logged_in"] = True
                st.session_state["user_application_id"] = st.session_state["application_id"]
                if login_attempt == 0:
                    st.session_state["require_password_change"] = True
                    update_login_attempts(st.session_state["application_id"])
                else:
                    st.session_state["require_password_change"] = False
                del st.session_state["password"]  # It's a good practice not to store the password longer than necessary.
                load_and_cache_json_data()
                st.experimental_rerun()  # Refresh the app state after successful login
            else:
                st.session_state["logged_in"] = False
                st.session_state["password_correct"] = False
        else:
            return

    # Display login form
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Your Best Candidate AI</h1>",
        unsafe_allow_html=True,
    )
    application_id = st.text_input("User email", type="default", key="application_id")
    password = st.text_input("Password", type="password", key="password")
    
    # Implement login button
    if st.button('Login'):
        password_entered()  # Call the function to check password when the button is clicked

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return st.session_state.get("logged_in", False)


if __name__ == "__main__":
    if not st.session_state.get("logged_in", False):
        if not check_password():
            st.stop()  # Do not continue if check_password is not True.
        else:
            st.session_state.logged_in = True
            main()
    else:
        if st.session_state.get("require_password_change", False):
            change_password_page()
            st.stop()  # Stop execution to prevent main UI from showing
        else:
            main()  # Directly go to the main page if already logged in
