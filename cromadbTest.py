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
from unstructured.partition.auto import partition
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import openpyxl
from io import BytesIO
import time




load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

m_client = get_mongo_client()

current_filepath = None

def text_embedding(text):
        response = openai.embeddings.create(model="text-embedding-ada-002", input=text[:8000])
        return response.data[0].embedding

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

genai.configure(api_key='AIzaSyD-ChpS0ja3bYlIGFuBqEuI_Nei23ehSFU')

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

def load_data(csv_filepath, temp=False):
    print('file_name ===>',csv_filepath, temp)
    
    df=pd.read_csv(csv_filepath)
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
            print('Using collection 2')
            collection2.add(
                documents=batch_docs,
                ids=batch_ids
            )
        else:
            print('Using collection 1')
            collection.add(
                documents=batch_docs,
                ids=batch_ids
            )
    
    df.to_csv(csv_filepath, index=False)
    # session['current_data_source'] = file_name
        


def load_json_data(csv_filepath, json_data, temp=False):
    start = time.time()

    try:
        df = pd.json_normalize(json_data, sep='.') if isinstance(json_data, list) else pd.DataFrame(json_data)
    except Exception as e:
        return {'success': False, 'message': f"Error converting JSON: {e}"}

    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    df.to_csv(csv_filepath, index=False, escapechar='\\')
    print(f"JSON saved to {csv_filepath}")

    load_data(csv_filepath, temp=temp)
    print("JSON loaded into ChromaDB")

    elapsed = round(time.time() - start, 2)
    return {'success': True, 'message': 'Loaded successfully', 'load_time_seconds': elapsed}


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
    print(" -- execute_query1 is called --")
    print(f"   - Query: {query}")
    print(f"   - User ID: {user_id}")
    print(f"   - Temp: {temp}")
    print(f"   - Continuation Token: {continuation_token}")
    mongo_results_str = ""
    try:
        if "how many" in query.lower():
            print(" inside how many query")
            # Generate CSV file from MongoDB results
            mongo_client = get_mongo_client()
            ses_data_collection = mongo_client['user_db']['SES_data']
            sample_document = ses_data_collection.find_one()
            sample_document = convert_objectid_to_str(sample_document)
    
            # Use GeminiAI to generate a MongoDB query filter based on the user query and the sample document
            sample_document_str = json.dumps(sample_document, indent=2)
            prompt = f"""
                Based on the following MongoDB document structure:
                {sample_document_str}

                Generate the most accurate MongoDB query filter for the following user query: '{query}'

                - Give me only the query filter in the response. No other text.
                - 
                - Use only the field names present in the sample document.
                - Consider semantic matches (e.g., “job title” matches Current Job Title or Job Title/Headline).
                - Prioritize exact and strong relevance matches over casual mentions.
                - For gender, location, job title, skills, nationality, experience, certifications, and language—match the correct field as per sample.
                - If the query includes ranking or limits (e.g., "top 3"), do not include $limit in the filter—just build the query filter only.
                - Return only the final MongoDB query filter as a valid JSON object.
                """
            messages = [
                {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently."},
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
            csv_file_path = f"tmp/{user_id}_results.csv"
            print(f" Step 1 CSV File path is : {csv_file_path} ")
            # Write results to CSV
            with open(csv_file_path, mode='w', newline='') as file:
                print(f" Step 2 writing in csv  ")
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
    print('job_desc===>',job_desc)
    if job_desc['documents'] != []:
        embedding_query = ''.join(job_desc['documents'])
    else:
        embedding_query = 'give me top candidates'
    
    vector = text_embedding(query)
    if temp:
        results = collection2.query(    
            query_embeddings=vector,
            n_results=4000,
            include=["documents"]
        )
    else:
        print('Using collection')
        results = collection.query(    
            query_embeddings=vector,
            n_results=4000,
            include=["documents"]
        )
    available_tokens_for_results = 400000 - len(query)  # Subtracting an estimated length for static text in the prompt
    results_str = "".join(str(item) for item in results['documents'][0])
    # print('results_str===>',results_str)
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
    print(" -- execute_query2 is called --")
    print(f"   - Query: {query}")
    print(f"   - User ID: {user_id}")
    print(f"   - Temp: {temp}")
    print(f"   - Continuation Token: {continuation_token}")

    try:
        if "how many" in query.lower():
            print('Executing "how many" query')
            # Generate CSV file from MongoDB results
            print('  - Getting MongoDB client')
            mongo_client = get_mongo_client()
            print('  - Accessing SES_data collection')
            ses_data_collection = mongo_client['user_db']['SES_data']
            print('  - Finding one sample document')
            sample_document = ses_data_collection.find_one()
            print('  - Converting ObjectId to string')
            sample_document = convert_objectid_to_str(sample_document)
            print('  - Sample document:', sample_document)

            # Use GeminiAI to generate a MongoDB query filter based on the user query and the sample document
            sample_document_str = json.dumps(sample_document, indent=2)
            prompt = f"Based on the following sample document:\n{sample_document_str}\nGenerate a MongoDB query filter for the user query: '{query}' and give me only query filter in the response no other text"
            print('  - GeminiAI prompt for query filter:', prompt)

            messages = [
                {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently"},
                {"role": "user", "parts": prompt}
            ]
            print('  - Sending messages to GeminiAI')
            response = model.generate_content(messages)
            print('  - GeminiAI response:', response.text)

            json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                print('  - Extracted JSON query filter:', json_str)
            else:
                json_str = '{}'
                print('  - No JSON found in GeminiAI response, using default: {}')

            # Extract the data from the SES_data collection based on json_str query filter
            print('  - Executing MongoDB find query with filter:', json.loads(json_str))
            results = ses_data_collection.find(json.loads(json_str))
            results = [{k: v for k, v in result.items() if not isinstance(v, ObjectId)} for result in list(results)]
            print('  - MongoDB query results (first 5):', results[:5])
            mongo_results_str = json.dumps(results)
            # Define the CSV file path
            csv_file_path = f"tmp/{user_id}_results.csv"
            print("  - Step 1 fetching -- csv_file_path --", csv_file_path)
            csv_data = []
            # Write results to CSV
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                print("  - Step 2 writing -- csv_file_path --", csv_file_path)
                writer = csv.writer(file, escapechar='\\')
                # Write header
                if results:
                    writer.writerow(results[0].keys())
                    # Write data rows
                    for result in results:
                        writer.writerow(result.values())
                        csv_data.append(result.values())
                    print('CSV file generated successfully')
                    yield results, 'csv'
                    return
                else:
                    print('No results found for the "how many" query, CSV not generated.')
                    yield [], 'stop' # Or some other appropriate finish signal
                    return
        else:
            mongo_results_str = None
    except Exception as e:
        print('Error in execute_query (handling "how many"):', e)

    try:
        print('Processing general query')
        job_desc = job_query.get('job_profile')
        if job_desc and job_desc.get('documents') != []:
            embedding_query = ''.join(job_desc['documents'])
            print('  - Using job description documents for embedding query:', embedding_query[:100] + '...')
        else:
            embedding_query = 'give me top 3 candidates'
            print('  - No job description documents found, using default embedding query:', embedding_query)

        print('  - Getting embedding for query:', embedding_query[:100] + '...')
        vector = get_embedding(embedding_query)

        if temp:
            print('  - Querying collection2 (temporary)')
            results = collection2.query(
                query_embeddings=vector,
                n_results=4000,
                include=["documents"]
            )
        else:
            print('  - Querying collection')
            results = collection.query(
                query_embeddings=vector,
                n_results=4000,
                include=["documents"]
            )
        print('  - ChromaDB query results (first result documents):', results.get('documents', [[]])[0][:100] if results.get('documents') else 'No results')

        available_tokens_for_results = 400000 - len(query)  # Subtracting an estimated length for static text in the prompt
        results_str = "".join(str(item) for item in results['documents'][0]) if results.get('documents') and results['documents'][0] else ""

        if mongo_results_str:
            single_line_text = mongo_results_str
            print('  - Using MongoDB results for prompt')
        else:
            single_line_text = results_str.replace("\n", " ")
            print('  - Using ChromaDB results for prompt')

        is_truncated = len(single_line_text) > available_tokens_for_results
        print('  - Is results text truncated?', is_truncated)
        if is_truncated:
            single_line_text = single_line_text[:available_tokens_for_results]  # Truncate results to fit within token limits
            print('  - Truncated results text:', single_line_text[:100] + '...')

        if continuation_token:
            # Adjust the prompt or setup to continue from where it left off
            prompt = f"Continuing : {continuation_token.replace('Context',single_line_text)} and answer the query : {query}"
            print('  - Using continuation token for prompt:', prompt[:200] + '...')
        else:
            # prompt = f'Here is the job description: {embedding_query}. Based on the resume data provided in {single_line_text}, please answer the following query: {query}. Ensure that your answer directly addresses the query and matches the job requirements and candidate information provided. Thank you!'
            prompt = f"""You are an AI assistant specialized in matching job candidates to job descriptions. Your task is to analyze the provided job description and candidate information, then answer specific queries about the candidate's suitability for the role. Please follow these guidelines:  Job Description: {embedding_query} Candidate Information: {single_line_text} Query: {query}  Instructions: 1. Carefully read and understand the job description and candidate information. 2. Focus on addressing the specific query provided. 3. Base your response solely on the information given in the job description and candidate details. 4. Provide a concise, relevant answer that directly addresses the query. 5. If the query is unclear or cannot be answered based on the given information, politely ask for clarification. 6. Do not make assumptions or infer information not explicitly stated in the provided data. 7. If greeting the user, respond appropriately without listing candidates.  Your response should be professional, unbiased, and tailored to the specific query and information provided. 8. If you dont have the job discription then answer the query based on the cadidate information"""
            print('  - Initial prompt for general query:', prompt[:200] + '...')

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
            print('  - Using specific prompt for truncated "how many" query with download link')

        # Start streaming
        print('  - Using chat completions for general query. 2')
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            stream=True
        )
        print('  - OpenAI response received, starting to yield chunks')

        # Yield each chunk as it is received
        for message in response:
            chunk = message.choices[0].delta.content if message.choices[0].delta.content is not None else ""
            finish_res = message.choices[0].finish_reason
            yield chunk, finish_res
            print('  - Yielded chunk:', chunk)
    except Exception as e:
        print('Error in execute_query2 (general query):', e)


# def execute_query3(query, user_id, temp=False, continuation_token=None, user_conversation=[]):
#     print(" -- execute_query is called --")
#     print(f"   - Query: {query}")
#     print(f"   - User ID: {user_id}")
#     print(f"   - Temp: {temp}")
#     print(f"   - Continuation Token: {continuation_token}")
#     print(f"   - Current File Path from Session: {session.get('current_file_path')}")
#     print(f"   - Current Data Source from Session: {session.get('current_data_source')}")

#     mongo_results_str = ""
#     csv_results_str = ""
#     fetched_from_csv = False

#     try:
#         # Dynamic query generation based on user input for MongoDB
#         if "dsasaasasd" in query.lower():
#             print("Inside how many query")
#             mongo_client = get_mongo_client()
#             ses_data_collection = mongo_client['user_db']['SES_data']
#             sample_document = ses_data_collection.find_one()
#             sample_document = convert_objectid_to_str(sample_document)
#             sample_document_str = json.dumps(sample_document, indent=2)
#             prompt = f"Based on the following sample document:\n{sample_document_str}\nGenerate a MongoDB query filter for the user query: '{query}' and give me only query filter in the response no other text"
#             messages = [
#                 {"role": "model", "parts": "You are an AI that generates MongoDB query filters based on user queries and sample documents. Use regex for search more efficiently"},
#                 {"role": "user", "parts": prompt}
#             ]
#             response = model.generate_content(messages)
#             json_match = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
#             json_str = json_match.group(1) if json_match else '{}'
#             results = ses_data_collection.find(json.loads(json_str))
#             results = [convert_objectid_to_str(result) for result in list(results)]
#             mongo_results_str = json.dumps(results)
#             csv_file_path = f"tmp/{user_id}_results.csv"
#             print(f"Step 1 CSV File path is: {csv_file_path}")
#             with open(csv_file_path, mode='w', newline='') as file:
#                 print("Step 2 writing in csv")
#                 writer = csv.writer(file, escapechar='\\')
#                 if results:
#                     writer.writerow(results[0].keys())
#                     for result in results:
#                         writer.writerow(result.values())

#         # Fetch from CSV if the current data source is CSV and a path exists
#         if session.get('current_data_source') == 'csv' and session.get('current_file_path'):
            
#             fetched_from_csv = True
#             csv_file_path = session.get('current_file_path')
#             try:
#                 with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
#                     reader = csv.DictReader(file)
#                     csv_data = list(reader)

#                 # Basic filtering (adapt as needed)
#                 # relevant_candidates = [
#                 #     candidate for candidate in csv_data
#                 #     if any(keyword.lower() in ' '.join(candidate.values()).lower() for keyword in query.lower().split())
#                 # ]

    
#                 # Or 

#                 # --- Improved Filtering Logic ---
#                 relevant_candidates = []
#                 if csv_data:
#                     query_lower = query.lower()
#                     nationality_keywords = [kw.strip() for kw in re.findall(r"(?:nationality:|are|who are|from)\s*([a-zA-Z]+)", query_lower)]
#                     language_keywords = [kw.strip() for kw in re.findall(r"(?:speak|speaks|language:|who speak)\s*([a-zA-Z]+)", query_lower)]
#                     skill_keywords = [kw.strip() for kw in re.findall(r"(?:skills:|have skills|with skills|skill:)\s*([a-zA-Z]+)", query_lower)]
#                     job_title_keywords = [kw.strip() for kw in re.findall(r"(?:role of|for the role of|job title:)\s*([a-zA-Z\s]+)", query_lower)]

#                     for candidate in csv_data:
#                         nationality_match = not nationality_keywords or any(kw.lower() in candidate.get("Nationality", "").lower() for kw in nationality_keywords) or any(kw.lower() in candidate.get("nationalities", "").lower() for kw in nationality_keywords) # Check both 'Nationality' and 'nationalities' if they exist
#                         language_match = not language_keywords or any(kw.lower() in candidate.get("Languages", "").lower() for kw in language_keywords) or any(kw.lower() in candidate.get("languages", "").lower() for kw in language_keywords) # Check both 'Languages' and 'languages' if they exist
#                         skill_match = not skill_keywords or any(kw.lower() in candidate.get("Skills", "").lower() for kw in skill_keywords) or any(kw.lower() in candidate.get("Skills", "").lower() for kw in skill_keywords) # Check 'Skills'
#                         job_title_match = not job_title_keywords or any(kw.lower() in candidate.get("Job Title", "").lower() for kw in job_title_keywords) or any(kw.lower() in candidate.get("jobTitle", "").lower() for kw in job_title_keywords) # Check both 'Job Title' and 'jobTitle' if they exist

#                         if nationality_match and language_match and skill_match and job_title_match:
#                             relevant_candidates.append(candidate)

#                 # Format relevant candidates for the prompt
#                 formatted_candidates = []
#                 for candidate in relevant_candidates:
#                     candidate_info = "\n".join(f"{key}: {value}" for key, value in candidate.items())
#                     formatted_candidates.append(candidate_info)

#                 csv_results_str = "\n\n".join(formatted_candidates)
#                 print("<><><><> Fetched from CSV (Session Path):\n", csv_results_str)

               

#             except FileNotFoundError:
#                 csv_results_str = "Uploaded CSV file not found."
#                 print("Error: Uploaded CSV file not found.")
#             except Exception as e:
#                 csv_results_str = f"Error reading uploaded CSV file: {e}"
#                 print(f"Error reading uploaded CSV file: {e}")

#         # Process the job description for embedding (for vector database)
#         job_desc = job_query.get('job_profile')
#         embedding_query = ''.join(job_desc['documents']) if job_desc['documents'] else 'give me top candidates'

#         # Generate embeddings for the query (for vector database)
#         vector = get_embedding(query)
#         vector_results = None
#         if temp and not fetched_from_csv: # Only query vector DB if not using CSV
#             print(' Query Using collection 2  ')
#             vector_results = collection2.query(query_embeddings=vector, n_results=4000, include=["documents"])
#         elif not fetched_from_csv:
#             print(' Query Using collection 1  ')
#             vector_results = collection.query(query_embeddings=vector, n_results=4000, include=["documents"])

#         vector_results_str = "".join(str(item) for item in vector_results['documents'][0]) if vector_results and vector_results['documents'] else ""

#         available_tokens_for_results = 400000 - len(query)  # Adjust for token limits

#         # Determine the candidate information to use in the prompt, prioritizing CSV
#         candidate_info_for_prompt = csv_results_str if fetched_from_csv else vector_results_str.replace("\n", " ")
#         is_truncated = len(candidate_info_for_prompt) > available_tokens_for_results
#         if is_truncated:
#             candidate_info_for_prompt = candidate_info_for_prompt[:available_tokens_for_results]

#         # Prepare the prompt for the AI model
#         if continuation_token:
#             prompt = f"Continuing: {continuation_token.replace('Context', candidate_info_for_prompt)} and answer the query: {query}"
#         else:
#             prompt = f"""
#                 You are required to act as a specialized expert in matching job candidates with job descriptions.
#                 Your task is to analyze the provided job description and candidate information, then answer specific queries regarding the candidate's suitability for the role.

#                 Job Description: {embedding_query}
#                 Candidate Information: {candidate_info_for_prompt}

#                 Guidelines for responses:
#                 1. Carefully read and understand both the job description and candidate information.
#                 2. If the query includes greetings, respond briefly in one line.
#                 3. Base your response on the provided job description and candidate details.
#                 4. Offer concise, relevant answers that address the query directly.
#                 5. If the query is unclear, ask the candidate for further clarification.
#                 6. Responses should be professional, unbiased, and tailored to the specific query and information provided.
#                 7. If the job description is missing, base your response solely on the candidate's information.
#                 8. If all information is missing, provide a general yet focused response related to job seeking or recruitment without highlighting the lack of details.
#                 9. Ensure the response is professional yet easy to understand.
#                 10. Always review the last few messages and base your reply on them.
#                 11. If candidate names are mentioned, use those and avoid introducing new ones, unless the user requests new candidates.
#                 12. Return the candidate information in the following structured format, describing only the available details:

#                     Candidate ID : [Candidate ID]
#                     Full Name: [Full Name]
#                     Job Title: [Job Title]
#                     email: [email]
#                     Skills: [List of Skills]
#                     Experience & Overview: [Summary of Experience]
#                     Location: [Location]
#                     languages: [languages]
#                     Nationality: [nationalities]

#                 Last conversation: {str(user_conversation)}
#                 Reply to this query: {query}
#             """

#         messages = [
#             {"role": "system", "content": "You answer questions about job candidates."},
#             {"role": "user", "content": prompt}
#         ]

#         # Start streaming the response
#         response = openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             temperature=0.1,
#             stream=True
#         )

#         # Yield each chunk as it is received
#         for message in response:
#             chunk = message.choices[0].delta.content if message.choices[0].delta.content is not None else ""
#             finish_res = message.choices[0].finish_reason
#             yield chunk, finish_res

#     except Exception as e:
#         print('Error in execute_query', e)
#         print("Error in execute_query")


def extract_structured_data_with_ai(candidates=[], query="", embedding_query=""):

    print(" =========================== Candiates in AI model ============= :::::  ", len(candidates))
   
    prompt = f"""
            You are an AI expert designed to process and display structured candidate information. You will receive a list of candidate profiles and a user query. Your job is to return well-formatted, structured candidate details based on the query, or ask for clarification if the query is ambiguous.
            
            Context:
            Candidates: {candidates}



            Your guidelines:
            1. Understand the user query and relate it to the candidate data.
            2. If the query includes greetings or small talk, respond briefly and politely in one line.
            3. Focus your response only on the given candidate information.
            4. If the query is unclear, ask the user to rephrase or clarify it instead of guessing. Give small    examples 
            5. If the job description is missing, rely only on candidate data.
            6. If no candidate data is present (i.e., the candidate list is empty or blank), respond generally about job matching or career advice. **Do not say "No candidate profiles found" if the list is not empty.**
            7. When users ask for more candidates (e.g., 'continue', 'give me more', 'I need more'), return the next unique candidates in the list.
            8. Consider 'reference' field as the candidate identifier (e.g., reference = candidate ID).

Remember:
- Only write "No candidate profiles found for this request." if the candidates list is completely empty.
- Ensure the data is displayed in a well-structured format on your website. Preserve paragraph spacing, headings, subheadings, bold case titles, and proper indentation for fields.
- Do not repeat candidates already shown in previous responses. Always return only **unique** candidates from the provided list.

    1.**Candidate ID :** [reference]
      **Full Name:** [Full Name]
      **Gender:** [Gender]
      **Job Title:** [Current Job Title or headlineLower]
      **Email:** [Email]
      **Location:** [Address City or County/Region or Country]
      **Languages:** [Leave blank if not available]
      **Nationality:** [Nationality]

Respond to this query: "{query}"
"""

    messages = [
        {"role": "system", "content": "You answer questions about job candidates and extract structured information."},
        {"role": "user", "content": prompt}
    ]

    # Stream response from OpenAI and yield chunks for streaming
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        stream=True
    )

    # for message in response:
    #     chunk = message.choices[0].delta.content if message.choices[0].delta.content else ""
    #     finish_reason = message.choices[0].finish_reason
    #     print("===chunk===",chunk)
    #     print("===finish_res===",finish_reason)
    #     yield chunk, finish_reason
    return response



def pdf_to_text(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

def scanned_pdf_to_text(file_path):
    try:
        pages = convert_from_path(file_path)
        text = ""
        for i, page in enumerate(pages):
            text += pytesseract.image_to_string(page)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def get_job_description_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

        return text

def generate_mongo_query_with_openai(sample_document,user_conversation, user_query, job_description = None):

    sample_document_str = json.dumps(sample_document, indent=2)
    print("----------------- sample document from DB : --------------- ", sample_document_str)

    prompt = f"""
        You are an expert assistant for converting user queries into MongoDB filters.

        Below is a MongoDB document structure:
        {sample_document}

        Job Description:
        {job_description}

        Current User Query:
        '{user_query}'

        Previous User Conversation:
        {user_conversation if user_conversation else "None"}

        Your task:
        Generate a valid MongoDB query filter (as a JSON object only) that best matches the user's query and provided Job Description's job title, using the Job Description and prior conversation context.

        **Instructions:**
        1. Return only the MongoDB query filter – no explanation.
        2. Use only field names from the document structure.
        3. Use $regex for flexible text fields like  headlineLower, job title, , location.
        4. If the user is asking for **more candidates** (e.g., "show more", "give me 10 more", "find 13 more candidates"):
        - Treat it as a continuation.
        - Extract the most recent filterable criteria (e.g., headlineLower , job title,) from the previous conversation.
        - Reuse those same criteria in the new filter.
        - Do not modify or expand criteria unless the current query specifies changes.
        5. Do NOT include "$limit" in the filter.
        6. match candidates with query and give sample MongoDB document structure headlineLower, 
        6. Interpret "recruiter" broadly to include titles such as "HR Officer and Recruiter", "Human Resources Coordinator/Recruiter","HR Executive", "Human Resource", "Talent Acquisition", "Recruitment Specialist", "HR Recruiter", "Technical Recruiter", "Staffing Consultant", "Sourcing Specialist", "Talent Partner", "Hiring Partner", "People Operations", "Campus Recruiter", "Executive Search Consultant", "Headhunter", "HR Manager", "Director of Talent Acquisition", "VP of People", "Chief People Officer", "CHRO", and other similar roles involved in hiring and talent acquisition.
    
        **Your output must only be the MongoDB filter in JSON.**
        """
    
    messages = [
        {"role": "system", "content": (
            "You are a MongoDB expert AI. Your task is to convert natural language queries into  MongoDB filter queries. "
        )},
        {"role": "user", "content": prompt}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1
    )
    ai_response_text = response.choices[0].message.content
    print(" ============== Raw AI Response: ============= ", ai_response_text)
    json_match = re.search(r'```json\n(.*?)```', ai_response_text, re.DOTALL)
    if json_match:
        mongodb_query_str = json_match.group(1)
        print(" ============== Generated MongoDB Query (Dict): ============= ", mongodb_query_str)
        return json.loads(mongodb_query_str)
       
    else:
        raise ValueError(f"Could not extract a valid JSON MongoDB query from the AI response: {ai_response_text}")

def is_continuation_query(query):
    keywords = ['more', 'next', 'another', 'additional', 'continue']
    return any(kw in query.lower() for kw in keywords)

def extract_candidate_count(query):
    """Extracts the number of candidates requested from the query in a more flexible way."""
    patterns = [
        r"(\d+)\s+(?:more|additional|other) (?:candidate|profile)s?",
        r"(?:show|give|need|find)\s+(?:me)?\s*(\d+)\s+(?:more|additional|other) (?:candidate|profile)s?",
        r"(?:show|give|need|find)\s+(?:me)?\s*(\d+)\s+(?:candidate|profile)s?",
        r"(\d+)\s+(?:candidate|profile)s?\s+(?:more|please)?",
        r"(?:I want|I'd like)\s+(\d+)\s+more",
        r"(?:get|fetch)\s+(\d+)\s+more"
        # Add more patterns as you identify common phrasing
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:

            return int(match.group(1))
    return None

def create_excel_file(data, filename="candidates.xlsx"):
    """
    Creates an Excel file from a list of dictionaries, handling nested dictionaries.
    """
    if not data:
        return None

    workbook = openpyxl.Workbook()
    sheet = workbook.active

    try:
        # Determine all unique headers, including keys from nested dictionaries
        all_headers = set()
        for row_data in data:
            all_headers.update(row_data.keys())
            for value in row_data.values():
                if isinstance(value, dict):
                    all_headers.update(value.keys())

        headers = list(all_headers)
        sheet.append(headers)

        # Write data
        for row_data in data:
            row = []
            for header in headers:
                value = row_data.get(header)
                if isinstance(value, dict):
                    # If the value is a dictionary, concatenate its key-value pairs into a string
                    cell_value = ", ".join(f"{k}: {v}" for k, v in value.items())
                    row.append(cell_value)
                else:
                    row.append(value if value is not None else "")
            sheet.append(row)

        # Save the workbook to a BytesIO object (in-memory file)
        excel_file = BytesIO()
        workbook.save(excel_file)
        excel_file.seek(0)  # Reset the file pointer to the beginning

        return excel_file

    except Exception as e:
        print(f"Error creating Excel file: {e}")
        return None  # Explicitly return None on error

def execute_query3(query, user_id, temp=False, continuation_token=None, user_conversation=[], current_data_source=None ):
    job_description = ''
    print(" -- execute_query3 is called --")
    print(f"   - Query: {query}")
    print(f"   - User ID: {user_id}")
    
    print(f"   - Temp: {temp}")
    print(f"   - Continuation Token: {continuation_token}")
    print(f"   - User Conversation: {user_conversation}")
    print(f"   - Current File Path from Session: ")
    print(f"   - Current Data Source from Session:")
    print(f"   - Current PDF File path from Session: ")
    print(f"   - useSES from Session: ")

    result = db.users.find_one(
        { "app_name": user_id },
        { "job_description": 1, "data_file": 1, "_id": 0 }
    )

    if result:
        job_description_path = result.get("job_description")

    print("============= job_description_path ===============  ",job_description_path )
    data_source = session.get('current_data_source')

    if data_source == 'json_upload':
        temp = True
    else:
        temp = False

    

    if job_description_path:
        job_description = get_job_description_from_file(job_description_path)
        # print(" job description ---\n\n ", job_description )
    else:
        None

    if not current_data_source:
        yield "No candidate data source available. Please wait a minute, or re-upload/select SES, CSV, or JSON.", "error"
        return


    
    response_text = ""
    candidates = []  # List to store candidate dictionaries.

    try:
        if current_data_source == 'SES':
            print(" ============== Processing with useSES ============= ")
            mongo_client = get_mongo_client()
            ses_data_collection = mongo_client['user_db']['SES_data']

            sample_document = ses_data_collection.aggregate([{"$sample": {"size": 1}}]).next()
            if sample_document:
                sample_document = convert_objectid_to_str(sample_document)
                # print(" -------- converted object to str ------------ ", sample_document)

                if is_continuation_query(query):
                    print(" === inside is continuation query :: === ")
                    mongodb_query = session.get('last_mongodb_query')
                    if not mongodb_query:
                        print("No stored filter for continuation. Falling back to new query.")
                        mongodb_query = generate_mongo_query_with_openai(sample_document, user_conversation, query, job_description)
                        session['pagination_offset'] = 0
                    else:
                        session['pagination_offset'] = session.get('pagination_offset', 0) + 20
                else:
                    mongodb_query = generate_mongo_query_with_openai(sample_document, user_conversation, query, job_description)
                    session['pagination_offset'] = 0

                session['last_mongodb_query'] = mongodb_query
                offset = session.get('pagination_offset', 0)

                requested_count  = extract_candidate_count(query)

                if requested_count:
                    
                    if requested_count > 20:
                        try:
                            print(f"Fetching all {requested_count} candidates...")
                            results = ses_data_collection.find(mongodb_query).limit(requested_count)
                            candidates = [{k: v for k, v in result.items() if not isinstance(v, ObjectId)} for result in list(results)]
                            print(f"Found {len(candidates)} candidates.")

                            csv_file_path = f"tmp/{user_id}_results.csv"
                            csv_data = []

                            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                                print("  - Step 2 writing -- csv_file_path --", csv_file_path)
                                writer = csv.writer(file, escapechar='\\')

                                if candidates:
                                    # Write the header row using the keys of the first candidate (dictionary)
                                    writer.writerow(candidates[0].keys())

                                    # Iterate through each candidate (dictionary) in the list
                                    for candidate in candidates:
                                        # Write the values of the current candidate
                                        writer.writerow(candidate.values())
                                        csv_data.append(list(candidate.values())) # Append the list of values

                                    print('CSV file generated successfully')
                                    yield candidates, 'csv'
                                    return
                                else:
                                    print('No results found for the "how many" query, CSV not generated.')
                                    yield ['CSV not generated'], 'stop' # Or some other appropriate finish signal
                                    return

                        except Exception as e:
                            print(f"Error : {e}")
                            return None     

                try:
                    print(f"Final MongoDB Query: {mongodb_query} with offset: {offset}")
                    results = ses_data_collection.find(mongodb_query).skip(offset).limit(20)
                    candidates = list(results)
                    print("------------------ result ========= \n\n", candidates)
                    print("- candidates -- ", len(candidates))
 
                    # Updated: yield streaming response
                    response = extract_structured_data_with_ai(candidates, query=query)

                    for message in response:
                        chunk = message.choices[0].delta.content if message.choices[0].delta.content else ""
                        finish_reason = message.choices[0].finish_reason
                        yield chunk, finish_reason

                    if not candidates:
                        response_text = "No matching candidates found in the database."
                        print(response_text)

                except Exception as e:
                    print(f"Error generating MongoDB query: {e}")
                    return
            else:
                response_text = "Sample document not found in SES_data collection."
                print(response_text)
                return

        if current_data_source == 'CSV' or current_data_source == 'JSON_UPLOAD':
            print(" ============== Processing CSV Data ============= ")
            fetched_from_csv = True
            db_result = db.users.find_one({"app_name": user_id}, {"data_file": 1, "_id": 0})

            if db_result:
                csv_file_path = db_result.get('data_file')

            print(" ==================== current csv file path :: == ", csv_file_path)
            try:
                with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    csv_data = list(reader)

                # Basic filtering (adapt as needed)
                # relevant_candidates = [
                #     candidate for candidate in csv_data
                #     if any(keyword.lower() in ' '.join(candidate.values()).lower() for keyword in query.lower().split())
                # ]

    
                # Or 

                # --- Improved Filtering Logic ---
                
                relevant_candidates = []
                if csv_data:
                    query_lower = query.lower()
                    nationality_keywords = [kw.strip() for kw in re.findall(r"(?:nationality:|are|who are|from)\s*([a-zA-Z]+)", query_lower)]
                    language_keywords = [kw.strip() for kw in re.findall(r"(?:speak|speaks|language:|who speak)\s*([a-zA-Z]+)", query_lower)]
                    skill_keywords = [kw.strip() for kw in re.findall(r"(?:skills:|have skills|with skills|skill:)\s*([a-zA-Z]+)", query_lower)]
                    job_title_keywords = [kw.strip() for kw in re.findall(r"(?:role of|for the role of|job title:)\s*([a-zA-Z\s]+)", query_lower)]

                    for candidate in csv_data:
                        nationality_match = not nationality_keywords or any(kw.lower() in candidate.get("Nationality", "").lower() for kw in nationality_keywords) or any(kw.lower() in candidate.get("nationalities", "").lower() for kw in nationality_keywords) # Check both 'Nationality' and 'nationalities' if they exist
                        language_match = not language_keywords or any(kw.lower() in candidate.get("Languages", "").lower() for kw in language_keywords) or any(kw.lower() in candidate.get("languages", "").lower() for kw in language_keywords) # Check both 'Languages' and 'languages' if they exist
                        skill_match = not skill_keywords or any(kw.lower() in candidate.get("Skills", "").lower() for kw in skill_keywords) or any(kw.lower() in candidate.get("Skills", "").lower() for kw in skill_keywords) # Check 'Skills'
                        job_title_match = not job_title_keywords or any(kw.lower() in candidate.get("Job Title", "").lower() for kw in job_title_keywords) or any(kw.lower() in candidate.get("jobTitle", "").lower() for kw in job_title_keywords) # Check both 'Job Title' and 'jobTitle' if they exist

                        if nationality_match and language_match and skill_match and job_title_match:
                            relevant_candidates.append(candidate)

                # Format relevant candidates for the prompt
                formatted_candidates = []
                for candidate in relevant_candidates:
                    candidate_info = "\n".join(f"{key}: {value}" for key, value in candidate.items())
                    formatted_candidates.append(candidate_info)

                csv_results_str = "\n\n".join(formatted_candidates)
                # print("<><><><> Fetched from CSV (Session Path):\n", csv_results_str)

               

            except FileNotFoundError:
                yield  "Uploaded CSV file not found.", "error"
                print("Error: Uploaded CSV file not found.")
            except Exception as e:
                yield  f"Error reading uploaded CSV file", "error"
                

            # Process the job description for embedding (for vector database)
            # job_desc = job_query.get('job_profile')
            # embedding_query = ''.join(job_desc['documents']) if job_desc['documents'] else 'give me top candidates'

            # Generate embeddings for the query (for vector database)
            vector = get_embedding(query)
            vector_results = None
            if temp : # Only query vector DB if not using CSV
                print(' Query Using collection 2  ')
                vector_results = collection2.query(query_embeddings=vector, n_results=4000, include=["documents"])
            else :
                print(' Query Using collection 1  ')
                vector_results = collection.query(query_embeddings=vector, n_results=4000, include=["documents"])
                # print("vector_results : ", vector_results)
                # print(" Line 1112 Number of results found:", vector_results["documents"])

            vector_results_str = "".join(str(item) for item in vector_results['documents'][0]) if vector_results and vector_results['documents'] else ""

            # print(" Line 1115 ======= vector_results_str === : ",vector_results_str )
            available_tokens_for_results = 400000 - len(query)  # Adjust for token limits

            # Determine the candidate information to use in the prompt, prioritizing CSV
            candidate_info_for_prompt = csv_results_str if fetched_from_csv else vector_results_str.replace("\n", " ")
            
            is_truncated = len(candidate_info_for_prompt) > available_tokens_for_results
            if is_truncated:
                
                candidate_info_for_prompt = candidate_info_for_prompt[:available_tokens_for_results]
                # print(" is_truncated : candidate_info_for_prompt:  ", candidate_info_for_prompt)

            # Prepare the prompt for the AI model
            if continuation_token:
                print(" Line 1128 ========= inside continuation token ========== : ", )
                prompt = f"Continuing: {continuation_token.replace('Context', candidate_info_for_prompt)} and answer the query: {query}"
            else:
                # print(" Line 1131 ========= Not  continuation token Direct ly prompt  ========== : ", )
                prompt = f"""
                    You are required to act as a specialized expert in matching job candidates with job descriptions.
                    Your task is to analyze the provided job description and candidate information, then answer specific queries regarding the candidate's suitability for the role.

                    Job Description: {job_description}
                    Candidate Information: {candidate_info_for_prompt}

                    Guidelines for responses:
                    1. Carefully read and understand both the job description and candidate information.
                    2. fetch candidates exect match with job description 
                    2. If the query includes greetings, respond briefly in one line.
                    3. If the query includes a language requirement, match it with the 'Languages' field in the candidate data. Consider variations in phrasing (e.g., "Arabic-speaking", "knows Hindi", "fluent in English") and match against standardized English language names only.
                    3. Base your response on the provided job description and candidate details.
                    4. Offer concise, relevant answers that address the query directly.
                    5. If the query is unclear, ask the candidate for further clarification.
                    6. Responses should be professional, unbiased, and tailored to the specific query and information provided.
                    7. If the job description is missing, base your response solely on the candidate's information.
                    8. If all information is missing, provide a general yet focused response related to job seeking or recruitment without highlighting the lack of details.
                    9. Ensure the response is professional yet easy to understand.
                    10. Always review the last few messages and base your reply on them.
                    11. If candidate names are mentioned, use those and avoid introducing new ones, unless the user requests new candidates.
                    12. Return the candidate information in the following structured format, describing only the available details:

                    Special Case – Best Candidate Comparison:
                        - If the query involves comparing candidates or selecting the best one:
                        - Compare based on Experience, Skills, and Education.
                        - Return the best matching candidate with a brief explanation.
                        - In case of a tie, mention multiple candidates and explain briefly.

                        Formatting Rules:
                        - If fewer than 3 candidates match, state the count explicitly.
                        - Present data in a structured, readable format for the web. Use proper paragraph spacing, bolded section titles, and indentations.

                    Remember
                    - *Fetch candidates only if their profile closely aligns with the job description title and core responsibilities.*
                    - *If the user explicitly asks to fetch all candidates, then return all candidates matching the job title, even if their profiles are not closely aligned with the full job description.*   
                    - *Avoid job title mismatches or overly broad interpretations.*  
                    - If candidate count is less than 3, state the count along with candidate details (e.g., “There are 2 candidates matching the job description”).   
                    - **All responses should be in a well-structured, professional document format, suitable for direct display or reporting.**  
                    - Ensure the data is displayed in a clean format: preserve paragraph spacing, headings, subheadings, bold field labels, and proper indentation.

                      1.**Candidate ID :** [reference]
                        **Full Name:** [Full Name]
                        **Gender:** [Gender]
                        **Job Title:** [Current Job Title]
                        **Email:** [Email]
                        **Skills:** [Skills]
                        **Location:** [Address City or County/Region or Country]
                        **Languages:** [Leave blank if not available]
                        **Nationality:** [Nationality]

                    Last conversation: {str(user_conversation)}
                    Reply to this query: {query}
                """

            messages = [
                {"role": "system", "content": "You answer questions about job candidates and extract and write in fix structured  information."},
                {"role": "user", "content": prompt}
            ]

            # Start streaming the response
            response = openai_client.chat.completions.create(
                model="gpt-4o",
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
        error_message = f"No candidates found in the database matching your query. "
        yield error_message, "error"  #  yield error

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

def clear_collection(user_id):
    collection.delete(where={"user_id": {"$eq": user_id}})
    collection2.delete(where={"user_id": {"$eq": user_id}})
    print("Collection cleared successfully for user_id:", user_id)