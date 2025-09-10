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
from datetime import datetime, date
import json
from mongo_connection import get_mongo_client
import google.generativeai as genai
import csv,re
from unstructured.partition.auto import partition
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import openpyxl
from io import BytesIO
import time
from typing import List, Dict, Any, Optional
from query_generator import generate_vector_query_and_fetch_results,format_chroma_results



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

def get_chat_history(user_id: str, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the chat history for a given user from MongoDB.
    If chat_id is provided, it filters by that specific chat_id.
    """
    query: Dict[str, Any] = {"user_id": user_id}
    if chat_id:
        query["chat_id"] = chat_id
    
    # In a real scenario, sort by a 'timestamp' field in descending order
    # For mock data, we just return the filtered list.
    history = list(chat_history_collection.find(query)) 
    
    # Sort history if it contains a 'timestamp' field in real data
    # history.sort(key=lambda x: x.get('timestamp', float('-inf')), reverse=True)
    
    return history


# ---------------- Ranking Utilities ----------------
def _parse_month_year(date_str: Optional[str]) -> Optional[datetime]:
    try:
        if not date_str:
            return None
        ds = str(date_str).strip()
        if ds.lower() in ("present", "current"):
            return datetime.utcnow()
        # Support formats like MM/YYYY or YYYY-MM
        for fmt in ("%m/%Y", "%Y-%m", "%Y/%m", "%b %Y", "%Y"):
            try:
                dt = datetime.strptime(ds, fmt)
                return dt
            except Exception:
                continue
        # Try DD/MM/YYYY
        for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(ds, fmt)
                return dt
            except Exception:
                continue
    except Exception:
        return None
    return None


def _months_since(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    now = datetime.utcnow()
    # Convert to months difference
    return max(0, (now.year - dt.year) * 12 + (now.month - dt.month))


def _extract_recency_months(experiences: Optional[List[Dict[str, Any]]]) -> int:
    if not experiences:
        return 120  # default: very old
    months_list = []
    for exp in experiences:
        try:
            if exp.get("is_current") or str(exp.get("end_date", "")).strip().lower() in ("present", "current"):
                months_list.append(0)
                continue
            end_dt = _parse_month_year(exp.get("end_date")) or _parse_month_year(exp.get("start_date"))
            months = _months_since(end_dt)
            if months is not None:
                months_list.append(months)
        except Exception:
            continue
    return min(months_list) if months_list else 120


def _compute_quality_score(doc: Dict[str, Any]) -> float:
    score = 0.0
    # Education weight
    education = doc.get("education") or []
    highest = 0.0
    for edu in education:
        degree = str(edu.get("degree", "")).lower()
        if any(k in degree for k in ["phd", "doctor"]):
            highest = max(highest, 1.0)
        elif any(k in degree for k in ["master", "msc", "m.s", "mtech", "m.tech"]):
            highest = max(highest, 0.7)
        elif any(k in degree for k in ["bachelor", "bsc", "b.tech", "btech", "b.e", "be"]):
            highest = max(highest, 0.5)
        elif degree:
            highest = max(highest, 0.3)
    score += 0.5 * highest  # up to 0.5

    # Certifications
    certs = doc.get("certifications") or []
    score += min(0.3, 0.03 * len(certs))  # up to 0.3

    # Awards / honors
    awards = doc.get("awards_honors") or []
    score += min(0.2, 0.05 * len(awards))  # up to 0.2

    return min(1.0, score)


def _compute_title_matched_years(doc: Dict[str, Any], context_text: Optional[str]) -> Optional[float]:
    try:
        if not context_text:
            return None
        title_years = doc.get("total_experience_by_title")
        if not isinstance(title_years, dict) or not title_years:
            return None
        ctx = str(context_text).lower()
        # Tokenize context a bit to allow partial matches
        ctx_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", ctx))
        matched_sum = 0.0
        for title, yrs in title_years.items():
            try:
                if not title or not isinstance(yrs, (int, float)):
                    continue
                t = str(title).lower()
                t_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", t))
                # Match if full title is substring or there is meaningful token overlap
                if t in ctx or bool(t_tokens & ctx_tokens):
                    matched_sum += float(yrs)
            except Exception:
                continue
        return matched_sum if matched_sum > 0 else None
    except Exception:
        return None


def compute_candidate_ranking(doc: Dict[str, Any], context_text: Optional[str] = None) -> float:
    """
    Score combines:
    - Time invested: total experience years
    - Recency: months since last role ended (current -> 0)
    - Quality: education, certifications, awards
    Returns score in [0, 1.0+]
    """
    # Time invested
    # Prefer title-matched experience based on job description/query context
    years = _compute_title_matched_years(doc, context_text)
    if years is None:
        total_years = doc.get("total_experience_years") or doc.get("total_years_of_experience")
        try:
            years = float(total_years) if total_years is not None else None
        except Exception:
            years = None
        if years is None and isinstance(doc.get("total_experience_by_title"), dict):
            try:
                years = float(sum(v for v in doc["total_experience_by_title"].values() if isinstance(v, (int, float))))
            except Exception:
                years = 0.0
    years = years or 0.0
    time_score = min(1.0, years / 10.0)  # cap at 10 years

    # Recency (smaller months -> better)
    months = _extract_recency_months(doc.get("experience"))
    # Map months to [0..1], 0 months -> 1.0, 60+ months -> ~0.0
    recency_score = max(0.0, 1.0 - (months / 60.0))

    # Quality
    quality_score = _compute_quality_score(doc)

    # Weighted sum
    score = 0.4 * time_score + 0.4 * recency_score + 0.2 * quality_score
    return score


def _extract_skill_patterns_from_mongo_query(query_obj: Any) -> List[str]:
    patterns: List[str] = []
    try:
        def walk(obj: Any, parent_key: Optional[str] = None):
            if isinstance(obj, dict):
                # If this dict represents a field filter on skills/key_qualifications
                for field in ("skills", "key_qualifications"):
                    if field in obj and isinstance(obj[field], dict):
                        inner = obj[field]
                        if "$regex" in inner and isinstance(inner["$regex"], str):
                            patterns.append(inner["$regex"])
                # Recurse all values
                for k, v in obj.items():
                    walk(v, k)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it, parent_key)
        walk(query_obj)
    except Exception:
        return []
    return patterns


def _compute_skills_overlap_count(doc: Dict[str, Any], regex_patterns: List[str]) -> int:
    if not regex_patterns:
        return 0
    try:
        skills_list = doc.get("skills") or []
        quals_list = doc.get("key_qualifications") or []
        summary = doc.get("summary") or ""
        # Safely normalize to strings
        def _to_text(items):
            try:
                return " \n ".join([str(x) for x in items if x])
            except Exception:
                return ""
        corpus = (" "+_to_text(skills_list)+" "+_to_text(quals_list)+" "+str(summary)).lower()
        match_count = 0
        seen = set()
        for pat in regex_patterns:
            try:
                compiled = re.compile(pat, re.IGNORECASE)
                if compiled.search(corpus):
                    if pat not in seen:
                        seen.add(pat)
                        match_count += 1
            except Exception:
                # If regex fails, fallback to substring search
                term = str(pat).split("|")[0].strip()
                if term and term.lower() in corpus and pat not in seen:
                    seen.add(pat)
                    match_count += 1
        return match_count
    except Exception:
        return 0

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
            ses_data_collection = mongo_client['user_db']['Updated_SES_data']
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
                - Consider semantic matches (e.g., "job title" matches Current Job Title or Job Title/Headline).
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


def extract_structured_data_with_ai(candidates=[], query="", embedding_query=""):

    print(" =========================== Candiates in AI model ============= :::::  ", len(candidates))
   
    prompt = f"""
You are an AI expert designed to process and display structured candidate information. Your primary goal is to take a given list of candidate profiles and present their details in a clear, well-formatted, and structured manner. You will also handle user queries and provide appropriate responses or seek clarification when needed.

**Context:**
List of Candidate Profiles to Display: {candidates}

**Your Guidelines:**
1.  **Output Goal:** Your main objective is to display the provided candidate profiles in the exact structured format specified below.
2.  **Handle Small Talk:** If the query includes greetings or brief small talk, respond politely and concisely (one line).
3.  **Focus on Data:** Restrict your response strictly to the provided candidate information. Do not fabricate or infer data.
5.  **Missing Job Description (from query):** If no job description or specific job role is provided in the user's query, your default action is to display *all* available candidate data from the `List of Candidate Profiles to Display` in the structured format below.
6.  **Empty Candidate List:**
    * If the `List of Candidate Profiles to Display` is completely empty or `[]`, respond generally about job matching or career advice (e.g., "It seems I don't have specific candidate profiles matching your current search. Would you like some general advice on optimizing your job search criteria?").
    * If the `List of Candidate Profiles to Display` is empty or `[]` AND the user's query is unclear or ambiguous, politely ask the user to rephrase or clarify. Provide small examples of what you can process (e.g., "Are you looking for candidates by location, job title, or something else? For example, 'Show candidates in London' or 'Who are the mechanical technicians?'").
    * **Crucially, do NOT say "No candidate profiles found" if the list provided is not empty.**
7.  **"More Candidates" Request:** If the user asks for more candidates (e.g., 'continue', 'give me more', 'I need more'), return the next unique candidates from the provided list that haven't been displayed yet.
8.  **Candidate Identifier:** The 'Candidate ID' field corresponds to the 'reference' field in the raw data.
9.  **Uniqueness:** Ensure you only return **unique** candidates. Do not repeat candidates from previous turns.

**Output Formatting:**
* Display candidate information in a clear, structured, and easy-to-read format.
* Preserve paragraph spacing, use headings/subheadings, bold titles, and proper indentation.
* **For each candidate, display only the fields that are available (not null, empty string, or empty list) using the following structure:**

    **Candidate ID :** [reference]
    **Full Name:** [Full Name]
    **Gender:** [Gender]
    **Job Title:** [Current Job Title or headline]
    **Email:** [Email]
    **Location:** [Address City or County/Region or Country]
    **Languages:** [knownLanguages]
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

def get_job_description_from_file(file_path, min_text_length=100):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        text = pdf_to_text(file_path)
        if len(text) < min_text_length:
            text = scanned_pdf_to_text(file_path)
        return text

    elif ext == ".txt":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"File read error: {e}")
            return ""
    
    else:
        print("Unsupported file type.")
        return ""

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def generate_mongo_query_with_openai(sample_document,user_conversation, user_query, job_description = None):

    
        
    sample_document_str = json.dumps(sample_document, indent=2, cls=CustomJSONEncoder)
    print("----------------- sample document from DB : --------------- ", sample_document_str)

    ses_nationality_instruction = """
        **Important:**
        For SES data source, candidates must be filtered to only those with nationality exactly "Iraqi". No foreign candidates should be included.
        """
    prompt = f"""
        ### Task:
        Generate a **MongoDB query filter** based on the job description and the candidate profile, considering the following flexible matching rules:

        Below is a MongoDB document structure:
        {sample_document}

        Job Description:
        {job_description}

        Current User Query:
        '{user_query}'

        Previous User Conversation:
        {user_conversation if user_conversation else "None"}

        ### Instructions:
        1. **Experience Matching:**
        - From the given job description, identify and extract only job titles that are directly relevant to the described role and responsibilities
        - Do not include unrelated job titles or roles that are outside the primary scope of the job description.
        - Use the field experience.title (not nested with $elemMatch).
        - Create a case-insensitive regular expression ($options: "i") that contains no more than 3–4 closely related job titles.
        - Related titles should include:
            - Variations and synonyms of the main role
            - Common alternative names for the position in industry usage
        -Avoid adding generic or tangential roles unless they are explicitly stated as acceptable in the job description.

        2. **Experience Years Matching:**
        - Create a separate filter for experience years using both `total_experience_years` and `experience.total_years` fields.
        - Use `$gte` operator for both fields in an `$or` condition.

        3. **Skills Matching:**
        - Match skills using `skills` and `key_qualifications` fields.
        - Use **detailed regular expressions** with specific skill patterns separated by pipes (|).
        - For skills, use patterns like: "Hazard.*Recognition|Risk.*Evaluation|Permit.*to.*Work|First.*Aid|Fire.*Fighting|H2S|Confined.*Space|Excavation.*Safety|Work.*at.*Height|Lifting.*Operations|Scaffolding.*Inspection|Control.*of.*hazardous.*energy|Chemical.*Hazards|Electrical.*Hazard|Defensive.*Driving"
        - For key_qualifications, use patterns like: "Hazard.*Identification|Risk.*Assessment|HSE.*Management|Safety.*Training|Emergency.*Response|Permit.*to.*Work|Regulatory.*Compliance"
        - Include the `$options: "i"` flag for case-insensitive matching.

        4. **Education Matching:**
        - Match **degrees** using `education.degree` field with `$regex` for patterns like "Engineering|Science|Petroleum Engineering".
        - Also include an alternative condition using `experience` with `$elemMatch` that checks `total_years` field with `$gte` for the minimum years.
        - Use `$or` to combine both education conditions.

        5. **Structure Requirements:**
        - Use a `$and` condition at the root level to combine ALL filter groups.
        - Each major filter group should be structured as separate objects in the `$and` array.
        - Do NOT include nationality filters unless explicitly mentioned in the job description.
        - Do NOT duplicate conditions.

        6. **Field Path Specifications:**
        - Use `experience.title` (not `experience` with `$elemMatch` for title matching)
        - Use `experience.total_years` and `total_experience_years` for experience years
        - Use `education.degree` for degree matching
        - Use `experience` with `$elemMatch` only when checking nested experience years

        ### Generate the MongoDB query filter **JSON**:
        - The filter should return a valid JSON query with **all matching conditions**.
        - Structure should have exactly 4 main filter groups in `$and`: experience titles, experience years, skills, and education.
        - Do not return any extra explanations or comments. Only return the **MongoDB JSON query**.
        - Ensure proper nesting and use `$or` conditions within each filter group appropriately.
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

        try:
            # CRITICAL FIX: Convert string to dictionary
            mongodb_query_dict = json.loads(mongodb_query_str)
            print(" ============== Generated MongoDB Query (Dict): ============= ", mongodb_query_dict)
            print(" ============== Query Type: ============= ", type(mongodb_query_dict))
            
            # Optional: Add nationality filter for SES if needed
            # mongodb_query_dict = apply_iraqi_filter_to_ses_query(mongodb_query_dict)
            
            return mongodb_query_dict  # Return dictionary, not string
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Problematic JSON string: {mongodb_query_str}")
            raise ValueError(f"Could not parse JSON MongoDB query: {e}")
       
    else:
        raise ValueError(f"Could not extract a valid JSON MongoDB query from the AI response: {ai_response_text}")
    
def apply_iraqi_filter_to_ses_query(mongo_query_filter):
    # The correct filter to check nationality inside 'nationalities' array of objects
    iraqi_filter = {"nationalities.name": "Iraqi"}

    if "$and" in mongo_query_filter:
        # Append the iraqi_filter to the existing $and list
        mongo_query_filter["$and"].append(iraqi_filter)
    else:
        # If there's already a nationality filter that is different or incompatible, 
        # it's safer to override with the iraqi_filter
        if "nationalities.name" in mongo_query_filter and mongo_query_filter["nationalities.name"] != "Iraqi":
            mongo_query_filter["nationalities.name"] = "Iraqi"
        else:
            # Combine existing filter with Iraqi nationality filter
            mongo_query_filter = {"$and": [mongo_query_filter, iraqi_filter]}
    return mongo_query_filter


def is_continuation_query(query):
    keywords = ['more', 'next', 'another', 'additional', 'continue']
    return any(kw in query.lower() for kw in keywords)

from typing import Optional

def extract_positive_integer(query_text: str) -> Optional[int]:

    if not isinstance(query_text, str):
        raise TypeError("Input 'query_text' must be a string.")
    number_pattern = re.compile(r'\b[-+]?\d+(?:\.\d+)?\b')

    extracted_strings = number_pattern.findall(query_text)

    for num_str in extracted_strings:
        try:
            # If it has a decimal point, it's a float; ignore it.
            if '.' in num_str:
                continue
            
            # Convert to integer
            num = int(num_str)

            # If it's positive (greater than 0), return it immediately.
            if num > 0:
                return num # Return the first found positive integer

        except ValueError:
            pass 

    return None # Return None if no positive integer is found


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
    print(f"   - Current Data Source from Session: ",current_data_source)

    print(f"   - useSES from Session: ")

    result = db.users.find_one(
        { "app_name": user_id },
        { "job_description": 1, "data_file": 1, "_id": 0 }
    )

    if result:
        job_description_path = result.get("job_description")

    # print("============= job_description_path ===============  ",job_description_path )
    data_source = session.get('current_data_source')

    if data_source == 'json_upload':
        temp = True
    else:
        temp = False

    

    if job_description_path:
        job_description = get_job_description_from_file(job_description_path)
        print(" job description ---\n\n ", job_description )
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
            ses_data_collection = mongo_client['user_db']['test_db']

            print(" ---------------- ses_data_collection ----------  ", ses_data_collection)

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

                requested_count  = extract_positive_integer(query)
                print(" Request candidates  before query  = === : ",requested_count )


                try:
                    results_list =[]

                    if not requested_count or requested_count == 0:
                        requested_count = 20

                    print(" Request candidates  after  query  = === : ",requested_count )
                    print(f"Final MongoDB Query: {mongodb_query} with offset: {offset}")
                    # Fetch a larger pool to rank globally, then paginate in-memory
                    fetch_limit = max((requested_count or 20) * 10, 100)
                    results_list = list(ses_data_collection.find(mongodb_query).limit(fetch_limit))

                    # Sort results by ranking before formatting
                    try:
                        ranking_context = f"{job_description or ''} {query or ''}".strip()
                        # Extract skill regex patterns from the stored MongoDB query for tie-breaking
                        mongo_skill_patterns = _extract_skill_patterns_from_mongo_query(mongodb_query)
                        # Composite sort: first by score desc, then by skills overlap desc
                        def _sort_key(r):
                            base_score = compute_candidate_ranking(r, ranking_context)
                            skills_hit = _compute_skills_overlap_count(r, mongo_skill_patterns)
                            # Return a tuple for sorting: primary score, secondary overlap
                            return (base_score, skills_hit)
                        results_list.sort(key=_sort_key, reverse=True)
                    except Exception:
                        pass
                    candidates = []
                    
                    if results_list:
                        # print("========== results_list recived  ============ ", results_list)
                        for result in results_list:
                            # print("========== Result ============ ", result)
                            # Reformat the output to match the desired keys
                            personal_info = result.get("personal_info", {}) or {}
                            full_name = personal_info.get("name")
                            email = personal_info.get("email")
                            nationality = personal_info.get("nationality")

                            candidate_data = {
                                "Candidate ID": result.get("candidate_id"),
                                "Full Name": full_name,
                                "Email": email,
                                # "Gender": result.get("gender"),
                                # "Job Title": result.get("headline"),
                                # "Email": result.get("personal_info.email"),
                                # "Location": result.get("address", {}).get("cityName"), # Handle nested field gracefully
                                "Languages": result.get("languages_spoken"),
                                # "Nationality": [nat.get("name") for nat in result.get("nationalities", [])] if result.get("nationalities") else [],
                                # "candidateId": result.get("candidateId")
                                "Total Experience": result.get("total_experience_years"),
                                # "Skills": result.get("skills"),
                                "Education": result.get("education"),     
                                "nationality" : nationality,
                                "Rank Score": round(compute_candidate_ranking(result, ranking_context), 3)
                            }
                            candidates.append(candidate_data)

                        if not candidates:
                            # Case 1: No candidates found after processing (even if raw results were there, they might have been filtered out by get())
                            yield "We searched our entire database, but couldn't find any candidates with those exact attributes. How about trying a similar role or related skills?", "error"
                            return # Exit the generator
                    else:
                        print("No candidates found for the query or at the current offset.")
                        response_ = "We searched our entire database, but couldn't find any candidates with those exact attributes. How about trying a similar role or related skills?"
                        yield response_ ,"error"
                        

                    # Paginate after ranking
                    start = offset or 0
                    end = (offset or 0) + (requested_count or 20)
                    paged_candidates = candidates[start:end]

                    if not paged_candidates:
                        yield "No more candidates available for the current criteria.", "stop"
                        return

                    print(f"=== Returning {len(paged_candidates)} ranked candidates (from pool {len(candidates)}) -- ")
                   

                    if not candidates:
                        yield "Could you please clarify what specific criteria you are looking for in the best matching candidates? For example, are you interested in candidates by location, job title, or specific skills?","error"

                    else:
                        total_candidates_count = len(paged_candidates)

                        yield f"Found {total_candidates_count} matching candidate(s):\n\n", None 

                        formatted_output_chunks = format_candidates_for_display(paged_candidates)
                        for chunk in formatted_output_chunks:
                            yield chunk, None # Yield chunks, assuming 'None' for finish_reason until the end
                        yield "", "stop" # Signal completion at the end

                except Exception as e:
                    print(f"Error generating MongoDB query: {e}")
                    response_ = "We searched our entire database, but couldn't find any candidates with those exact attributes. How about trying a similar role or related skills?"
                    yield response_ ,"error"
            else:
                response_text = "Sample document not found in SES_data collection."
                print(response_text)
                yield response_text, "error1"
                return

        if current_data_source == 'CSV' or current_data_source == 'JSON_UPLOAD':
            print(" ============== Processing CSV Data ============= ")
            fetched_from_csv = True
            db_result = db.users.find_one({"app_name": user_id}, {"data_file": 1, "_id": 0})

            if db_result:   
                csv_file_path = db_result.get('data_file')

            # print(" ==================== current csv file path :: == ", csv_file_path)
            try:
                with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    csv_data = list(reader)

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
                yield  f"Error reading uploaded CSV or JSON file", "error"
                

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
                print(" - Total documents in collection:", collection.count())
                
                vector_results = collection.query(query_embeddings=vector, n_results=200, include=["documents"])

            vector_results_str = "".join(str(item) for item in vector_results['documents'][0]) if vector_results and vector_results['documents'] else ""


            # print(" Line 1115 ======= vector_results_str === : ",vector_results_str )
            available_tokens_for_results = 128000 - len(query)  # Adjust for token limits

            # Determine the candidate information to use in the prompt, prioritizing CSV
            candidate_info_for_prompt = csv_results_str if fetched_from_csv else vector_results_str.replace("\n", " ")
            
            print(" Line 1265 ======= candidate_info_for_prompt === : ",len(candidate_info_for_prompt) )

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
                You are a specialized assistant for matching job candidates to job descriptions using CSV-style candidate data.

                Your task is to evaluate candidates against the provided job description and respond to user queries based on that evaluation.

                ---
                **Job Description:**
                {job_description if job_description else "None provided"}

                **Candidate Information:**
                {candidate_info_for_prompt} 

                **User Query:**
                {query}

                **Previous Conversation:**
                {str(user_conversation) if user_conversation else "None"}
                ---

                ### Matching Rules:

                1. **Strictly match candidates by Job Title** from the JD. Match only those whose job title **exactly or closely aligns** with the JD title.
                - Ignore unrelated titles unless user says "show all."
                2. If the JD includes core responsibilities or tools, match them with candidate skills as secondary criteria.
                3. If the query mentions a **language**, check if it's included in the candidate's "Languages" field (match known variations like "knows Hindi", "fluent in English").
                4. If the query is a **greeting or casual**, reply briefly and politely in one line.
                5. If the query involves **finding best** or **comparing** candidates, compare strictly based on:
                - **Experience (years)**
                - **Skills**
                - **Education** (if available)

                ### Formatting Guidelines:

                - Display candidates only if they pass filters.
                - Show results in this exact structured format (omit empty fields):

                    **Candidate ID:** [reference]  
                    **Full Name:** [Full Name]  
                    **Gender:** [Gender]  
                    **Job Title:** [Current Job Title]  
                    **Email:** [Email]  
                    **Skills:** [Skills]  
                    **Location:** [Address City / County / Region / Country]  
                    **Languages:** [Languages]  
                    **Nationality:** [Nationality]  

                - Use proper paragraph spacing, bold field labels, and indentation.
                - If < 3 candidates match, state that clearly. If none, say so politely.

                ### Behavior Rules:

                - If JD is missing, use only candidate data to respond.
                - If all data is missing, give a general recruitment-related response.
                - Never return broad or loosely related profiles unless user explicitly asks.
                - Ask for clarification if the query is ambiguous.

                **Output only the final structured candidate response or direct reply. No comments. No markdown.**
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
        import traceback
        traceback.print_exc()
        error_message = f"We searched our entire database, but couldn't find any candidates with those exact attributes. How about trying a similar role or related skills? "
        print(e)
        yield error_message, "error"  #  yield error

def format_candidates_for_display(candidates_list):
    for i, candidate in enumerate(candidates_list):
        output = []
        output.append(f"{i + 1}. ")

        if candidate.get("Candidate ID"):
            output.append(f"**Candidate ID:** {candidate['Candidate ID']}")
        if candidate.get("Full Name"):
            output.append(f"**Full Name:** {candidate['Full Name']}")
        if candidate.get("Email"):
            output.append(f"**Email:** {candidate['Email']}")
        if candidate.get("Languages"):
            try:
                # Handle both list of strings and list of dicts
                if isinstance(candidate["Languages"], list):
                    if all(isinstance(v, dict) and "language" in v for v in candidate["Languages"]):
                        langs = [v.get("language") for v in candidate["Languages"] if v.get("language")]
                    else:
                        langs = [str(v) for v in candidate["Languages"] if v]
                    output.append(f"**Languages:** {', '.join(langs)}")
            except Exception:
                output.append("**Languages:** ")
        if candidate.get("Total Experience") is not None:
            output.append(f"**Total Experience:** {candidate['Total Experience']}")
        if candidate.get("Education"):
            try:
                if isinstance(candidate["Education"], list):
                    edus = []
                    for edu in candidate["Education"]:
                        degree = edu.get("degree", "")
                        institution = edu.get("institution", "")
                        if degree and institution:
                            edus.append(f"{degree} ({institution})")
                        elif degree:
                            edus.append(degree)
                        elif institution:
                            edus.append(institution)
                    output.append(f"**Education:** {', '.join(edus)}")
            except Exception:
                output.append("**Education:** ")
        if candidate.get ("nationality"):
            output.append(f"**nationality:** {candidate['nationality']}")
        if candidate.get("Rank Score") is not None:
            try:
                score_val = candidate.get("Rank Score")
                numeric_val = float(score_val)
                score_percent = int(round(numeric_val * 100))
                score_str = f"{score_percent}"
            except Exception:
                score_str = str(candidate.get("Rank Score"))
            output.append(f"**Rank Score:** {score_str}")
        if candidate.get("Candidate ID"):
            # Generate profile URL using candidateId
            profile_url = f"https://secure.recruitly.io/candidate?id={candidate['Candidate ID']}"
            output.append(f"**Profile Link:** <a href='{profile_url}' target='_blank' style='color:#1a0dab; text-decoration:underline;'>View Full Profile</a>")
        yield output[0] + '\n'.join(output[1:]) + "\n\n"


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