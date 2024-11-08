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
from datetime import datetime
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

# collection = client.get_or_create_collection("candidates",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
# collection2 = client.get_or_create_collection("candidates2",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
# job_query = client.get_or_create_collection("job_query",embedding_function=openai_ef)
collection = db['candidates']
collection2 = db['candidates2']
job_query = db['job_query']
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

def load_pdf_data(text, user_id):
    
    # Add code here to load the extracted text into the collection
    job_query.insert_one(
        {"text": text, "user_id": user_id, "date": datetime.strftime(datetime.now().date(), '%Y-%m-%d')}
    )

def load_data(file_name, user_id, temp=False):
    # Read CSV file
    df = pd.read_csv(file_name)
    
    # Convert DataFrame to list of dictionaries
    documents = []
    for _, row in df.iterrows():
        # Convert row to dictionary and add user_id and temp fields
        document = row.to_dict()
        document['user_id'] = user_id
        document['temp'] = temp
        
        documents.append(document)
    
    # Delete existing documents for the user
    if temp:
        collection2.delete_many({"user_id": user_id})
        collection2.insert_many(documents)
    else:
        collection.delete_many({"user_id": user_id})
        collection.insert_many(documents)
    
    # Convert DataFrame to JSON and save it
    json_file_path = file_name.replace('.csv', '.json')
    df.to_json(json_file_path, orient='records', indent=4)
    print(f"Data saved to {json_file_path}")


def load_json_data(json_data, user_id, file=False):
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
        load_data(csv_file_path, user_id, temp=True)
    else:
        load_data(csv_file_path, user_id, temp=False)
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
        print('query',query)
        if "how many" in query.lower():
            print('how many')
            # Generate CSV file from MongoDB results
            mongo_client = get_mongo_client()
            ses_data_collection = mongo_client['user_db']['SES_data']
            sample_document = ses_data_collection.find_one()
            sample_document = convert_objectid_to_str(sample_document)
            print('sample_document',sample_document)
            # Use GeminiAI to generate a MongoDB query filter based on the user query and the sample document
            sample_document_str = json.dumps(sample_document, indent=2)
            prompt = f"Based on the following sample document:\n{sample_document_str}\nGenerate a MongoDB query filter for the user query: '{query}' and give me only query filter in the response no other text"

            messages = [
                {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently"},
                {"role": "user", "parts": prompt}
            ]

            response = model.generate_content(messages)
            print('response',response.text)
            json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
            print('json_match',json_match)
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
        return 'Error in execute_query',e
    
    try:
        if len(mongo_results_str) == 0:
            if temp:
                print('temp')
                collection_to_use = collection2
            else:
                print('not temp')
                collection_to_use = collection
            # Get a sample document to understand the structure
            sample_document = collection_to_use.find_one()
            sample_document = convert_objectid_to_str(sample_document)
            # Use GeminiAI to generate a MongoDB query filter
            sample_document_str = convert_to_plain_text(sample_document)
            do = """{'address': {'$regex': '84'}}"""

            do_not = """{'type': 'CANDIDATE', 'address.postCode': {'$regex': '84'}}"""
            print(sample_document_str)
            prompt = f"""Based on the following sample document:
    {sample_document_str}

    Generate a MongoDB search query filter for: '{query}' that searches for specific columns using all needed mongo operators. give me json format must not use any other text

    Guidelines:
    1. For address,languages,skills,nationalities use regex to search directly like 'Iraq', 'Arabic', 'Python', 'English'
    2. For gender use $eq operator like 'Male' or 'Female'
    3. For age use $gte and $lte operators like '20' or '20-30'
    4. For experience use $gte and $lte operators like '2' or '2-3'
    5. Don't do this: {do_not} and do this: {do}
    6. Never use type: CANDIDATE in query filter
    7. Don't use limit in query filter
    8. Don't use json formatter for any field like address.postCode use regex instead
    9. Don't generate the query filter out of the {query}.
    10. Don't oversight just generate the query filter that user asked for
    """



            messages = [
                {
                    "role": "model",
                    "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for efficient searching.."
                },
                {
                    "role": "user",
                    "parts": prompt
                }
            ]

            response = model.generate_content(messages)
            print(response.text)
            # Extract the query filter from the response
            json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
            if json_match:
                query_filter = json.loads(json_match.group(1))
            else:
                query_filter = {}
            print(query_filter)
            # Execute the MongoDB query
            results = collection_to_use.find(query_filter).limit(4000)
            results = [convert_objectid_to_str(result) for result in list(results)]
            # print(results)
            if len(results) > 0:
                results_str = "".join(str(item) for item in results)
            else:
                results_str = "No results found"
        else:
            print('mongo_results_str')
            results_str = mongo_results_str

        job_desc = job_query.find_one({'user_id': user_id, 'date': datetime.strftime(datetime.now().date(), '%Y-%m-%d')})
        available_tokens_for_results = 400000 - len(query)  # Subtracting an estimated length for static text in the prompt
        single_line_text = mongo_results_str if len(mongo_results_str) >= available_tokens_for_results else results_str.replace("\n", " ")
        
        is_truncated = len(single_line_text) > available_tokens_for_results
        if is_truncated:
            single_line_text = single_line_text[:available_tokens_for_results]  # Truncate results to fit within token limits

        if continuation_token:
            # Adjust the prompt or setup to continue from where it left off
            prompt = f"Continuing : {continuation_token.replace('Context',single_line_text)} and answer the query : {query}"
        else:
            # prompt = f'Here is the job description: {embedding_query}. Based on the resume data provided in {single_line_text}, please answer the following query: {query}. Ensure that your answer directly addresses the query and matches the job requirements and candidate information provided. Thank you!'
            e_query = job_desc['text'] if job_desc else 'Unknown'
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
                        4. Structure your response in a clear, organized format:
                           - Use headers (##) for main sections
                           - Use bullet points (*) for listing items
                           - Use bold (**) for emphasis on important points
                           - Use tables for comparing multiple candidates
                        5. For candidate listings:
                           - Include a clear heading
                           - List each candidate with their key details in a structured format
                           - Highlight matching skills or qualifications
                        6. For statistical queries:
                           - Present numbers clearly at the beginning
                           - Follow with a brief explanation if needed
                           - Use bullet points for breakdowns
                        7. For comparison queries:
                           - Use a table format with clear columns
                           - Include relevant metrics for comparison
                        8. Keep responses concise and focused on the query
                        9. If the job description is missing, base your response solely on the candidate's information
                        10. If you get 'No results found', clearly state this and explain why no matches were found
                        11. For candidate references:
                            - Always include their full name and ID if available
                            - Format as: "John Doe (ID: JD123)"
                        12. End responses with a clear conclusion or recommendation when appropriate
                        13. Do not include the request in the response; only provide the answer.

                        Reply to this query: {query}

                        Note: Ensure all responses are properly formatted in Markdown and maintain professional language throughout.
                        """

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
                {"role": "user", "content": "What is your default response?"}
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
    except Exception as e:
        import traceback
        print('Error in execute_query',e)
        print(traceback.format_exc())
        pass





