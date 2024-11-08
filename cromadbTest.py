import pandas as pd
from dotenv import load_dotenv
import os
from flask import session, current_app
import openai
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json
from mongo_connection import get_mongo_client
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
import google.generativeai as genai
import csv,re

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

m_client = get_mongo_client()

def text_embedding(text):
        response = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return response["data"][0]["embedding"]

openai_client = OpenAI()
def get_embedding(text):
   text = text.replace("\n", " ")
   return openai_client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding

client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )

model = genai.GenerativeModel('gemini-1.5-flash')

genai.configure(api_key='AIzaSyAIUXWwE1Rd6vQgq7N9JJ1-8mkjSJln21Q')

db = m_client['user_db']

collection = client.get_or_create_collection("candidates",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
collection2 = client.get_or_create_collection("candidates2",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
job_query = client.get_or_create_collection("job_query",embedding_function=openai_ef)
chat_history_collection = db['chat_history']

def add_chat_message(user_id, message, response, chat_id, message_id):
    """
    Adds or updates a chat message and its response in the MongoDB collection.
    Checks if a message with the given message_id exists, updates it if available, or adds it if not.
    """
    timestamp = datetime.now().isoformat()
    document = {
        "user_id": user_id,
        "chat_id": chat_id,
        "message": message,
        "message_id": message_id,
        "response": response,
        "timestamp": timestamp
    }
    # Using update_one with upsert=True to update if exists, or insert if not
    chat_history_collection.update_one(
        {"message_id": message_id, "chat_id": chat_id},  # Filter by message_id
        {"$set": document},          # Update or set the document fields
        upsert=True                  # Insert as a new document if not exists
    )

def get_chat_history(user_id, chat_id=None):
    """
    Retrieves the chat history for a given user from MongoDB.
    If chat_id is provided, it filters by that specific chat_id.
    """
    query = {"user_id": user_id}
    if chat_id:
        query["chat_id"] = chat_id
    
    history = list(chat_history_collection.find(query).sort("timestamp", -1))
    return history

def get_chat_list(user_id):
    """
    Retrieves a list of chats for a given user, showing the first question of each chat,
    sorted by the timestamp of the first message in each chat to ensure consistent ordering.
    """
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"timestamp": 1}},
        {"$group": {
            "_id": "$chat_id", 
            "first_question": {"$first": "$message"},
            "first_timestamp": {"$first": "$timestamp"}  # Capture the timestamp of the first message
        }},
        {"$sort": {"first_timestamp": -1}},  # Ensure consistent ordering based on the first message's timestamp
        {"$project": {"chat_id": "$_id", "first_question": 1, "_id": 0}}
    ]
    return list(chat_history_collection.aggregate(pipeline))

def load_pdf_data(text):
    
    # Add code here to load the extracted text into the collection
    job_query.add(
        documents=[text],
        # metadatas=["job_profiles"],
        ids=["job_profile"]
    )

def load_data(file_name, temp=False):
    df=pd.read_csv(file_name)
    df.head()
    
    df['text'] = df.apply(lambda row: '\n'.join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    
    docs=df["text"].tolist() 
    ids= [str(x) for x in df.index.tolist()]
    # Define maximum batch size
    max_batch_size = 166
    
    # Splitting the documents and ids into batches and adding them to the collection
    for i in range(0, len(docs), max_batch_size):
        batch_docs = docs[i:i + max_batch_size]
        batch_ids = ids[i:i + max_batch_size]
        if temp:
            collection2.add(
                documents=batch_docs,
                ids=batch_ids
            )
        else:
            collection.add(
                documents=batch_docs,
                ids=batch_ids
            )
    if file_name:
        df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")


def load_json_data(json_data, file=False):
    """
    Converts JSON data to a DataFrame, then generates a new DataFrame with a 'text' column
    that concatenates all column names and data, and finally stores this data into Cromadb.
    Before storing into Cromadb, it saves the DataFrame as a CSV file in the 'csvdata' folder with the name 'json_data.csv'.

    Args:
    - json_data: A list of dictionaries, where each dictionary represents a document to be stored in Cromadb.
    """
    # Convert JSON data to DataFrame
    df = pd.DataFrame(json_data)

    # Save the DataFrame to a CSV file in the 'csvdata' folder
    csv_file_path = 'csvdata/json_data.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Ensure the directory exists
    df.to_csv(csv_file_path, index=False, escapechar='\\')
    print(f"JSON data saved to {csv_file_path} successfully.")
    
    if not file:
        load_data(csv_file_path, temp=True)
    else:
        load_data(csv_file_path, temp=False)
    print("JSON data loaded into Cromadb successfully.")

# def execute_query(query, user_id, temp=False):
#     job_desc = job_query.get('job_profile')

#     if job_desc['documents'] != []:
#         embedding_query = ''.join(job_desc['documents'])
#     else:
#         embedding_query = 'give me top 3 candidates'
    
#     vector = text_embedding(embedding_query)
#     if temp:
#         results = collection2.query(    
#             query_embeddings=vector,
#             n_results=1000,
#             include=["documents"]
#         )
#     else:
#         results = collection.query(    
#             query_embeddings=vector,
#             n_results=1000,
#             include=["documents"]
#         )

#     available_tokens_for_results = 100000 - len(query) - 200  # Subtracting an estimated length for static text in the prompt

#     # Convert results to string and truncate if necessary
#     results_str = "\n".join(str(item) for item in results['documents'][0])
#     if len(results_str) > available_tokens_for_results:
#         results_str = results_str[:available_tokens_for_results] 
#     prompt = f'```{results_str}```Based on the data in ```, answer {query}'

#     messages = [
#         {"role": "system", "content": "Welcome to BestCandidate AI Bot! I am here to answer your questions in a structured format. Please note that I will always respond in Markdown format. Let's get started!"},
#         {"role": "user", "content": prompt}
#     ]

#     # Start streaming
#     response = openai.ChatCompletion.create(
#         model="gpt-4-turbo-2024-04-09",
#         messages=messages,
#         temperature=0,
#         stream=True
#     )

#     # Collecting chunks
#     collected_chunks = []
#     collected_messages = []
#     for chunk in response:
#         collected_chunks.append(chunk)  # save the event response
#         if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0]:
#             chunk_message = chunk['choices'][0]['delta']['content']  # extract the message
#             collected_messages.append(chunk_message)  # save the message

#     # Combine collected messages into a single response
#     final_response = ''.join(collected_messages)
#     return final_response

from bson import ObjectId
def convert_objectid_to_str(doc):
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, dict):
                convert_objectid_to_str(value)
            elif isinstance(value, list):
                for item in value:
                    convert_objectid_to_str(item)
    return doc

def convert_to_plain_text(doc, indent=0):
    plain_text = ""
    for key, value in doc.items():
        if isinstance(value, dict):
            plain_text += " " * indent + f"{key}:\n" + convert_to_plain_text(value, indent + 2)
        elif isinstance(value, list):
            plain_text += " " * indent + f"{key}:\n"
            for item in value:
                if isinstance(item, dict):
                    plain_text += convert_to_plain_text(item, indent + 2)
                else:
                    plain_text += " " * (indent + 2) + f"- {item}\n"
        else:
            plain_text += " " * indent + f"{key}: {value}\n"
    return plain_text

def execute_query(query, user_id, temp=False, continuation_token=None, user_conversation = []):
    mongo_results_str = ""
    try:
        if "how many" in query.lower():
            # Generate CSV file from MongoDB results
            mongo_client = get_mongo_client()
            ses_data_collection = mongo_client['user_db']['SES_data']
            sample_document = ses_data_collection.find_one()
            sample_document = convert_objectid_to_str(sample_document)

            # Use GeminiAI to generate a MongoDB query filter based on the user query and the sample document
            sample_document_str = json.dumps(sample_document, indent=2)
            prompt = f"Based on the following sample document:\n{sample_document_str}\nGenerate a MongoDB query filter for the user query: '{query}' and give me only query filter in the response no other text"

            messages = [
                {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently"},
                {"role": "user", "parts": prompt}
            ]

            response = model.generate_content(messages)

            json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = '{}'
                
            #extract the data from the SES_data collcetion based on json_str query filter
            results = ses_data_collection.find(json.loads(json_str))
            results = [convert_objectid_to_str(result) for result in list(results)]
            mongo_results_str = json.dumps(results)
            # Define the CSV file path
            csv_file_path = f"/tmp/{user_id}_results.csv"
            
            # Write results to CSV
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file, escapechar='\\')
                # Write header
                writer.writerow(results[0].keys())
                # Write data rows
                for result in results:
                    writer.writerow(result.values())
    except Exception as e:
        print('Error in execute_query',e)
        print("error in execute_query")

    job_desc = job_query.get('job_profile')
    if job_desc['documents'] != []:
        embedding_query = ''.join(job_desc['documents'])
    else:
        embedding_query = 'give me top candidates'
    
    vector = get_embedding(embedding_query)
    if temp:
        results = collection2.query(    
            query_embeddings=vector,
            n_results=4000,
            include=["documents"]
        )
    else:
        results = collection.query(    
            query_embeddings=vector,
            n_results=4000,
            include=["documents"]
        )
    available_tokens_for_results = 400000 - len(query)  # Subtracting an estimated length for static text in the prompt
    results_str = "".join(str(item) for item in results['documents'][0])
    
    single_line_text = mongo_results_str if len(mongo_results_str) >= available_tokens_for_results else results_str.replace("\n", " ")
    
    is_truncated = len(single_line_text) > available_tokens_for_results
    if is_truncated:
        single_line_text = single_line_text[:available_tokens_for_results]  # Truncate results to fit within token limits

    if continuation_token:
        # Adjust the prompt or setup to continue from where it left off
        prompt = f"Continuing : {continuation_token.replace('Context',single_line_text)} and answer the query : {query}"
    else:
        # prompt = f'Here is the job description: {embedding_query}. Based on the resume data provided in {single_line_text}, please answer the following query: {query}. Ensure that your answer directly addresses the query and matches the job requirements and candidate information provided. Thank you!'
        e_query = "True" if 'give me top candidates' == embedding_query else embedding_query
        # prompt = f"""You are an AI assistant specialized in matching job candidates to job descriptions. Your task is to analyze the provided job description and candidate information, then answer specific queries about the candidate's suitability for the role. Please follow these guidelines:  Job Description: {e_query} Candidate Information: {single_line_text} Query: {query}  Instructions: 1. Carefully read and understand the job description and candidate information. 2. Focus on addressing the specific query provided. 3. Base your response solely on the information given in the job description and candidate details. 4. Provide a concise, relevant answer that directly addresses the query. 5. If the query is unclear or cannot be answered based on the given information, politely ask for clarification. 6. Do not make assumptions or infer information not explicitly stated in the provided data. 7. If greeting the user, respond appropriately without listing candidates.  Your response should be professional, unbiased, and tailored to the specific query and information provided. 8. If you dont have the job discription then answer the query based on the cadidate information 9. If all information is missing: Offer a general, professional response related to job seeking or recruitment without mentioning the lack of specific details. 10. If job description is missing: Focus on candidate qualifications without mentioning the lack of job details"""
        prompt = f"""
                    You are required to act as a specialized expert in matching job candidates with job descriptions.
                    Your task is to analyze the provided job description and candidate information, then answer specific queries regarding the candidate's suitability for the role.

                    Job Description: {e_query}
                    Candidate Information: {single_line_text}

                    Guidelines for responses:

                    1. Carefully read and understand both the job description and candidate information.
                    2. If the query includes greetings, respond briefly in one line.
                    3. Base your response on the provided job description and candidate details.
                    4. Offer concise, relevant answers that address the query directly.
                    5. If the query is unclear, ask the candidate for further clarification.
                    6. Responses should be professional, unbiased, and tailored to the specific query and information provided.
                    7. If the job description is missing, base your response solely on the candidate's information.
                    8. If all information is missing, provide a general yet focused response related to job seeking or recruitment without highlighting the lack of details.
                    9. Ensure the response is professional yet easy to understand.
                    10. Always review the last few messages and base your reply on them.
                    11. Always review the last few messages and base your reply on them. If candidate names are mentioned, use those and avoid introducing new ones, unless the user requests new candidates.
                    
                    last conversation: {str(user_conversation)}
                    Reply to this query: {query}"""

    messages = [
        # {"role": "system", "content": "You answer questions BestCandidate AI Bot. You will always answer in structured format and in markdown format and please don't use markdown word in response"},
        # {"role": "system", "content": "Welcome to BestCandidate AI Bot! I am here to answer your questions ensuring no repetitions in a structured format. Please note that I will always respond in Markdown format. Let's get started!"},
        {"role": "system", "content": """Welcome to YourBestCandidateAI!

                We're excited to partner with you in revolutionizing your hiring process by harnessing the power of AI to precisely match your job descriptions with the most suitable candidates. Here's how we ensure seamless collaboration to find your perfect candidates:

                Overview:
                At YourBestCandidateAI, we specialize in intelligently matching job descriptions with candidate resumes to pinpoint the ideal fit for your organization. With our advanced algorithms, we prioritize accuracy and efficiency to streamline your recruitment journey.

                How It Works:
                Detailed Job Descriptions:

                Provide thorough job descriptions, outlining specific roles, required skills, experience levels, educational qualifications, and any other pertinent criteria.
                Example: "Seeking a seasoned Software Engineer (Ref ID: SE123) with expertise in Python, cloud computing, and a minimum of 5 years of industry experience."
                Candidate Matching and Comparison:

                Request the best candidates based on your job description parameters, ensuring to include their reference IDs.
                Directly compare multiple candidates to identify the best fit for your organization.
                Receive curated lists of top candidates based on your specified criteria, each with their reference IDs included.
                Sample Requests:
                Finding Candidates:

                "Identify top candidates for the role of Marketing Manager (Ref ID: MM456)."
                "List candidates proficient in project management and finance, each with their respective reference IDs."
                Candidate Comparison:

                "Compare the qualifications of Jane Doe (Ref ID: JD789) and John Smith (Ref ID: JS987) for the Marketing Manager position, focusing on their digital marketing expertise and campaign management experience."
                "Provide a side-by-side analysis of candidates with over a decade of data analysis experience, ensuring to include their reference IDs."
                Candidate Count:

                "How many candidates match the criteria for the Data Scientist position?"
                "What's the total number of candidates with expertise in machine learning, and could you please provide their reference IDs?"
                Tips for Optimal Results:
                Detailed Descriptions: The more detailed your job descriptions, the better we can match candidates to your specific requirements.
                Consistent Naming/IDs: To ensure accurate responses, please refer to candidates consistently by their names or reference IDs.
                Additional Resources:
                For additional assistance or inquiries, feel free to reach out at any time. We're committed to your satisfaction and dedicated to helping you find the perfect candidate for your team.
                """},
        {"role": "user", "content": prompt}
    ]
    if 'how many' in query.lower() and is_truncated:
        download_link = f"https://yourbestcandidate.ai/download_csv/{user_id}_results.csv"
        messages = [
        # {"role": "system", "content": "You answer questions BestCandidate AI Bot. You will always answer in structured format and in markdown format and please don't use markdown word in response"},
        # {"role": "system", "content": "Welcome to BestCandidate AI Bot! I am here to answer your questions ensuring no repetitions in a structured format. Please note that I will always respond in Markdown format. Let's get started!"},
        {"role": "system", "content": f"""Welcome to YourBestCandidateAI!

                Just answer : "We have large data and because of token limitation I am not able to respond, but here is the download button of that data in CSV: [Download CSV]({download_link})" on user query 'how many'
                """},
            {"role": "user", "content": prompt}
        ]

    # Start streaming        
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        stream=True
    )

    # Yield each chunk as it is received
    for message in response:
        # print('message===>',message)
        chunk = message.choices[0].delta.content if message.choices[0].delta.content is not None else ""
        finish_res = message.choices[0].finish_reason 
        yield chunk,finish_res

def execute_query2(query, user_id, temp=False, continuation_token=None):

    try:
        if "how many" in query.lower():
            print('Executing "how many" query')
            # Generate CSV file from MongoDB results
            mongo_client = get_mongo_client()
            ses_data_collection = mongo_client['user_db']['SES_data']
            sample_document = ses_data_collection.find_one()
            sample_document = convert_objectid_to_str(sample_document)

            # Use GeminiAI to generate a MongoDB query filter based on the user query and the sample document
            sample_document_str = json.dumps(sample_document, indent=2)
            prompt = f"Based on the following sample document:\n{sample_document_str}\nGenerate a MongoDB query filter for the user query: '{query}' and give me only query filter in the response no other text"

            messages = [
                {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently"},
                {"role": "user", "parts": prompt}
            ]

            response = model.generate_content(messages)

            json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = '{}'
                
            # Extract the data from the SES_data collection based on json_str query filter
            results = ses_data_collection.find(json.loads(json_str))
            results = [{k: v for k, v in result.items() if not isinstance(v, ObjectId)} for result in list(results)]
            mongo_results_str = json.dumps(results)
            # Define the CSV file path
            csv_file_path = f"/tmp/{user_id}_results.csv"
            csv_data = []
            # Write results to CSV
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, escapechar='\\')
                # Write header
                writer.writerow(results[0].keys())
                # Write data rows
                for result in results:
                    writer.writerow(result.values())
                    csv_data.append(result.values())
            print('CSV file generated successfully')
            yield results, 'csv'
            return
        else:
            mongo_results_str = None
    except Exception as e:
        print('Error in execute_query:', e)
    try:
        job_desc = job_query.get('job_profile')
        if job_desc['documents'] != []:
            embedding_query = ''.join(job_desc['documents'])
        else:
            embedding_query = 'give me top 3 candidates'
        vector = get_embedding(embedding_query)
        if temp:
            results = collection2.query(    
                query_embeddings=vector,
                n_results=4000,
                include=["documents"]
            )
        else:
            results = collection.query(    
                query_embeddings=vector,
                n_results=4000,
                include=["documents"]
            )
        available_tokens_for_results = 400000 - len(query)  # Subtracting an estimated length for static text in the prompt
        results_str = "".join(str(item) for item in results['documents'][0])
        if mongo_results_str:
            single_line_text = mongo_results_str
        else:
            single_line_text = results_str.replace("\n", " ")
        is_truncated = len(single_line_text) > available_tokens_for_results
        if is_truncated:
            single_line_text = single_line_text[:available_tokens_for_results]  # Truncate results to fit within token limits
        if continuation_token:
            # Adjust the prompt or setup to continue from where it left off
            prompt = f"Continuing : {continuation_token.replace('Context',single_line_text)} and answer the query : {query}"
        else:
            # prompt = f'Here is the job description: {embedding_query}. Based on the resume data provided in {single_line_text}, please answer the following query: {query}. Ensure that your answer directly addresses the query and matches the job requirements and candidate information provided. Thank you!'
            prompt = f"""You are an AI assistant specialized in matching job candidates to job descriptions. Your task is to analyze the provided job description and candidate information, then answer specific queries about the candidate's suitability for the role. Please follow these guidelines:  Job Description: {embedding_query} Candidate Information: {single_line_text} Query: {query}  Instructions: 1. Carefully read and understand the job description and candidate information. 2. Focus on addressing the specific query provided. 3. Base your response solely on the information given in the job description and candidate details. 4. Provide a concise, relevant answer that directly addresses the query. 5. If the query is unclear or cannot be answered based on the given information, politely ask for clarification. 6. Do not make assumptions or infer information not explicitly stated in the provided data. 7. If greeting the user, respond appropriately without listing candidates.  Your response should be professional, unbiased, and tailored to the specific query and information provided. 8. If you dont have the job discription then answer the query based on the cadidate information"""
            

        messages = [
            {"role": "system", "content": """Welcome to YourBestCandidateAI!

                    We're excited to partner with you in revolutionizing your hiring process by harnessing the power of AI to precisely match your job descriptions with the most suitable candidates. Here's how we ensure seamless collaboration to find your perfect candidates:

                    Overview:
                    At YourBestCandidateAI, we specialize in intelligently matching job descriptions with candidate resumes to pinpoint the ideal fit for your organization. With our advanced algorithms, we prioritize accuracy and efficiency to streamline your recruitment journey.

                    How It Works:
                    Detailed Job Descriptions:

                    Provide thorough job descriptions, outlining specific roles, required skills, experience levels, educational qualifications, and any other pertinent criteria.
                    Example: "Seeking a seasoned Software Engineer (Ref ID: SE123) with expertise in Python, cloud computing, and a minimum of 5 years of industry experience."
                    Candidate Matching and Comparison:

                    Request the best candidates based on your job description parameters, ensuring to include their reference IDs.
                    Directly compare multiple candidates to identify the best fit for your organization.
                    Receive curated lists of top candidates based on your specified criteria, each with their reference IDs included.
                    Sample Requests:
                    Finding Candidates:

                    "Identify top candidates for the role of Marketing Manager (Ref ID: MM456)."
                    "List candidates proficient in project management and finance, each with their respective reference IDs."
                    Candidate Comparison:

                    "Compare the qualifications of Jane Doe (Ref ID: JD789) and John Smith (Ref ID: JS987) for the Marketing Manager position, focusing on their digital marketing expertise and campaign management experience."
                    "Provide a side-by-side analysis of candidates with over a decade of data analysis experience, ensuring to include their reference IDs."
                    Candidate Count:

                    "How many candidates match the criteria for the Data Scientist position?"
                    "What's the total number of candidates with expertise in machine learning, and could you please provide their reference IDs?"
                    Tips for Optimal Results:
                    Detailed Descriptions: The more detailed your job descriptions, the better we can match candidates to your specific requirements.
                    Consistent Naming/IDs: To ensure accurate responses, please refer to candidates consistently by their names or reference IDs.
                    Additional Resources:
                    For additional assistance or inquiries, feel free to reach out at any time. We're committed to your satisfaction and dedicated to helping you find the perfect candidate for your team.
                    """},
            {"role": "user", "content": prompt}
        ]

        if 'how many' in query.lower() and is_truncated:
            download_link = f"https://yourbestcandidate.ai/download_csv/{user_id}_results.csv"
            messages = [
            {"role": "system", "content": f"""Welcome to YourBestCandidateAI!

                    Just answer : "We have large data and because of token limitation I am not able to respond, but here is the download button of that data in CSV: [Download CSV]({download_link})" on user query 'how many'
                    """},
                {"role": "user", "content": prompt}
            ]

        # Start streaming        
        print('Using chat completions for general query. 2')
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            stream=True
        )

        # Yield each chunk as it is received
        for message in response:
            chunk = message.choices[0].delta.content if message.choices[0].delta.content is not None else ""
            finish_res = message.choices[0].finish_reason 
            yield chunk, finish_res
    except Exception as e:
        print('Error in execute_query2:', e)

def cromadb_test(file_name,query):    
    df=pd.read_csv(file_name)
    df.head()
    

    df['text'] = (
        'Candidate ID: ' + df['Candidate Id'].astype(str) + '\n' +
        'Employer Name: ' + df['Employer Name'].astype(str) + '\n' +
        'Start Date: ' + df['Start Date'].astype(str) + '\n' +
        'Country of birth: ' + df['Country of birth'].astype(str) + '\n' +
        'Marketing Emails: ' + df['Marketing Emails'].astype(str) + '\n' +
        'Current Job Title: ' + df['Current Job Title'].astype(str) + '\n' +
        'First Name: ' + df['First Name'].astype(str) + '\n' +
        'Gender: ' + df['Gender'].astype(str) + '\n' +
        'Email: ' + df['Email'].astype(str) + '\n' +
        'Home Phone: ' + df['Home Phone'].astype(str) + '\n' +
        'Candidate Owner: ' + df['Candidate Owner'].astype(str) + '\n' +
        'Status: ' + df['Status'].astype(str) + '\n' +
        'Optout SMS: ' + df['Optout SMS'].astype(str) + '\n' +
        'Internal Note: ' + df['Internal Note'].astype(str) + '\n' +
        'Address Country: ' + df['Address Country'].astype(str) + '\n' +
        'Expected Salary: ' + df['Expected Salary'].astype(str) + '\n' +
        'Old Candidate ID: ' + df['Old Candidate ID'].astype(str) + '\n' +
        'Current Company: ' + df['Current Company'].astype(str) + '\n' +
        'Expected Max Salary: ' + df['Expected Max Salary'].astype(str) + '\n' +
        'Preferred Sectors: ' + df['Preferred Sectors'].astype(str) + '\n' +
        'Address County/Region: ' + df['Address County/Region'].astype(str) + '\n' +
        'Last Contacted: ' + df['Last Contacted'].astype(str) + '\n' +
        'Preferred Job Titles: ' + df['Preferred Job Titles'].astype(str) + '\n' +
        'LinkedIn: ' + df['LinkedIn'].astype(str) + '\n' +
        'National Insurance Number: ' + df['National Insurance Number'].astype(str) + '\n' +
        'Candidate Skills: ' + df['Candidate Skills'].astype(str) + '\n' +
        'Education Level: ' + df['Education Level'].astype(str) + '\n' +
        'Current Salary: ' + df['Current Salary'].astype(str) + '\n' +
        'Driving License: ' + df['Driving License'].astype(str) + '\n' +
        'Job Types: ' + df['Job Types'].astype(str) + '\n' +
        'Rating: ' + df['Rating'].astype(str) + '\n' +
        'Preferences: ' + df['Preferences'].astype(str) + '\n' +
        'Address Line 1: ' + df['Address Line 1'].astype(str) + '\n' +
        'Created On: ' + df['Created On'].astype(str) + '\n' +
        'Position 1: ' + df['Position 1'].astype(str) + '\n' +
        'Position 2: ' + df['Position 2'].astype(str) + '\n' +
        'Current City: ' + df['Current City'].astype(str) + '\n' +
        'Date of Birth: ' + df['Date of Birth'].astype(str) + '\n' +
        "Father's Name: " + df["Father's Name"].astype(str) + '\n' +
        'Annual Leave Days: ' + df['Annual Leave Days'].astype(str) + '\n' +
        'Job Title/Headline: ' + df['Job Title/Headline'].astype(str) + '\n' +
        'Position 3: ' + df['Position 3'].astype(str) + '\n' +
        'Position 4: ' + df['Position 4'].astype(str) + '\n' +
        'Nationality: ' + df['Nationality'].astype(str) + '\n' +
        'Expected Min Salary: ' + df['Expected Min Salary'].astype(str) + '\n' +
        'Address - PIN/Postcode: ' + df['Address - PIN/Postcode'].astype(str) + '\n' +
        'Overview: ' + df['Overview'].astype(str) + '\n' +
        'Current Country: ' + df['Current Country'].astype(str) + '\n' +
        'Tags: ' + df['Tags'].astype(str) + '\n' +
        'City of Birth: ' + df['City of Birth'].astype(str) + '\n' +
        'Candidate Category: ' + df['Candidate Category'].astype(str) + '\n' +
        'End Date: ' + df['End Date'].astype(str) + '\n' +
        'Current JobType: ' + df['Current JobType'].astype(str) + '\n' +
        'Available From: ' + df['Available From'].astype(str) + '\n' +
        'Full Name: ' + df['Full Name'].astype(str) + '\n' +
        'Gender.1: ' + df['Gender.1'].astype(str) + '\n' +
        'Modified On: ' + df['Modified On'].astype(str) + '\n' +
        'Date of Birth.1: ' + df['Date of Birth.1'].astype(str) + '\n' +
        'Work Phone: ' + df['Work Phone'].astype(str) + '\n' +
        'Address Line 2: ' + df['Address Line 2'].astype(str) + '\n' +
        'Surname: ' + df['Surname'].astype(str) + '\n' +
        'Alternate Email Address: ' + df['Alternate Email Address'].astype(str) + '\n' +
        'University Degree: ' + df['University Degree'].astype(str) + '\n' +
        'Nationality.1: ' + df['Nationality.1'].astype(str) + '\n' +
        'Relocate: ' + df['Relocate'].astype(str) + '\n' +
        'Current Job Title.1: ' + df['Current Job Title.1'].astype(str) + '\n' +
        'Marketing SMS: ' + df['Marketing SMS'].astype(str) + '\n' +
        'Address City: ' + df['Address City'].astype(str) + '\n' +
        'Availability: ' + df['Availability'].astype(str) + '\n' +
        'Current Salary.1: ' + df['Current Salary.1'].astype(str) + '\n' +
        'Reason For Leaving: ' + df['Reason For Leaving'].astype(str) + '\n' +
        'Preferred Industries: ' + df['Preferred Industries'].astype(str) + '\n' +
        'Title: ' + df['Title'].astype(str) + '\n' +
        'Sick Leave Days: ' + df['Sick Leave Days'].astype(str) + '\n' +
        'Conversation Thread: ' + df['Conversation Thread'].astype(str) + '\n' +
        'Twitter: ' + df['Twitter'].astype(str) + '\n' +
        'Mobile: ' + df['Mobile'].astype(str)
    )

    
    def text_embedding(text):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )


    client = chromadb.Client()
    collection = client.get_or_create_collection("candidates",embedding_function=openai_ef)

    docs=df["text"].tolist() 
    ids= [str(x) for x in df.index.tolist()]


    collection.add(
        documents=docs,
        ids=ids
    )
    vector=text_embedding("give me top 5 candidate list")
    results=collection.query(    
        query_embeddings=vector,
        n_results=15,
        include=["documents"]
    )
    res = "\n".join(str(item) for item in results['documents'][0])    
    prompt=f'```{res}```Based on the data in ```, answer {query}'
    messages = [
            {"role": "system", "content": "You answer questions about 95th Oscar awards."},
            {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]    
    return response_message
