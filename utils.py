from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import boto3

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

def get_password(application_id):
    try:
        response = requests.get(
            f"http://vaibhavsharma3070.pythonanywhere.com/get_password?app_name={application_id}"
        )
        if response.status_code == 200:
            login_attempt = response.json().get('login', 0)  # Using .get() to avoid KeyError
            password = response.json().get('password', "Password not found")
            if password == "Password not found":
                return "Invalid application_id"
            return password, login_attempt
        else:
            return "😕 Invalid Email!"
            return None, 0  # Return None for password and 0 for login_attempt in case of error
    except requests.exceptions.JSONDecodeError:
        return "😕 Invalid Email!"

def send_reset_password_mail(to_email, subject, username, reset_link):
    from_email = "vaibhavsharma3070@gmail.com"
    password = "dariqpkhrhjldxpb"

    html_message = f"""
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Your Password</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                margin-top: 50px;
                background-color: #fff;
                border-radius: 10px;
                padding: 40px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
            }}
            p {{
                color: #666;
                margin-bottom: 30px;
            }}
            .btn {{
                display: inline-block;
                background-color: #007bff;
                color: #fff;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                margin-top: 20px;
                transition: background-color 0.3s ease;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
            .footer {{
                margin-top: 30px;
                color: #999;
                font-size: 14px;
            }}
            .username {{
                color: #007bff;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reset Your Password, <span class="username">{username}</span>!</h1>
            <p>You recently requested to reset your password. Click the button below to reset it:</p>
            <a href="{reset_link}" class="btn">Reset Password</a>
            <p>If you didn't request a password reset, you can ignore this email.</p>
            <div class="footer">
                <p>If you encounter any issues, please contact support at <a href="mailto:support@yourbestcandidate.ai">support@yourbestcandidate.ai</a>.</p>
                <p>This email was sent automatically. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(html_message, 'html'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()

def send_email(recipients, subject, username, reset_link):
    ses = boto3.client(
        'ses',
        region_name="us-east-1",
        aws_access_key_id="AKIAQ3EGVT7ALFGZHVAW",
        aws_secret_access_key="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"
    )

    html_message = f"""
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Your Password</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                margin-top: 50px;
                background-color: #fff;
                border-radius: 10px;
                padding: 40px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
            }}
            p {{
                color: #666;
                margin-bottom: 30px;
            }}
            .btn {{
                display: inline-block;
                background-color: #007bff;
                color: #fff;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                margin-top: 20px;
                transition: background-color 0.3s ease;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
            .footer {{
                margin-top: 30px;
                color: #999;
                font-size: 14px;
            }}
            .username {{
                color: #007bff;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reset Your Password, <span class="username">{username}</span>!</h1>
            <p>You recently requested to reset your password. Click the button below to reset it:</p>
            <a href="{reset_link}" class="btn">Reset Password</a>
            <p>If you didn't request a password reset, you can ignore this email.</p>
            <div class="footer">
                <p>If you encounter any issues, please contact support at <a href="mailto:support@yourbestcandidate.ai">support@yourbestcandidate.ai</a>.</p>
                <p>This email was sent automatically. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

    sender = "support@yourbestcandidate.ai"

    ses.send_email(
        Source=sender,
        Destination={'ToAddresses': recipients},
        Message={
            'Subject': {'Data': subject},
            'Body': {
                'Html': {'Data': html_message}
            }
        }
    )