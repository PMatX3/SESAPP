from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

# def handle_userinput(user_question):
#     #st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def calculate_total_tokens(conversation):
    return sum(len(message.content.split()) for message in conversation)

def manage_conversation(conversation, new_message, token_limit=4096):
    # Ensure conversation is a list
    if conversation is None:
        conversation = []
    
    total_tokens = calculate_total_tokens(conversation + [new_message])
    st.write(total_tokens)

    while total_tokens > token_limit and conversation:
        conversation.pop(0)
        total_tokens = calculate_total_tokens(conversation)

    conversation.append(new_message)
    return conversation

def handle_userinput(user_question):
    # The code here represents your existing functionality to get a response and update chat history
    response = st.session_state.conversation({'question': user_question})
    new_message = response['chat_history'][-1]  # Assuming the latest message is at the end

    # Manage conversation to ensure token limit is not exceeded
    updated_conversation = manage_conversation(st.session_state.chat_history, new_message)

    # Update the chat history in the session state
    st.session_state.chat_history = updated_conversation
    
    # Display the conversation
    for i, message in enumerate(updated_conversation):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    # Update the chat history in the session state
    st.session_state.chat_history = updated_conversation


def main():
    load_dotenv()
    #st.set_page_config(page_title="Find Your Candidate", page_icon="🇺🇸", layout="wide")
    st.set_page_config(page_title="Find Your Candidate", page_icon="Iraq", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Find Your Candidate")
    user_question = st.text_input("Ask a question about a candidate:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}","Hello BestCandidate AI"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human from SES."), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs of your resume/CVs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
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