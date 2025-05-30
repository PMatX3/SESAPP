import openai
from openai import OpenAI
import json 
import os
from dotenv import load_dotenv


load_dotenv() 

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_vector_query_and_fetch_results(user_conversation, user_query, job_description=None):

    prompt = f"""
You are an expert assistant for converting user queries into VectorDB filters.

Job Description:
{job_description or "None"}

Current User Query:
'{user_query}'

Previous User Conversation:
{user_conversation if user_conversation else "None"}

Your task:
Generate a cleaned search query (just a short phrase or keywords) that can be used to perform semantic similarity search in a vector database (e.g., ChromaDB) to fetch relevant candidate profiles.

**Instructions:**
1. Return only the cleaned query string (e.g., "fleet manager", "senior recruiter from India").
2. DO NOT return JSON or explanations — just the refined search string.
3. Use $regex for flexible text fields like  headlineLower, job title, , location
3. Use information from the Job Description and User Query to derive the search intent.
4. If the user is asking for **more candidates** (e.g., "show more", "give me 10 more", "find 13 more candidates"):
    - Treat it as a continuation.
    - Extract the most recent filterable criteria (e.g., headlineLower , job title,) from the previous conversation.
    - Reuse those same criteria in the new filter.
    - Do not modify or expand criteria unless the current query specifies changes.
5. Include terms like "Fleet Manager", "Technical Recruiter", "Human Resource", "VP of People", etc. if the user’s query implies those roles.
6. Interpret "recruiter" broadly to include titles like:
   - "HR Officer and Recruiter"
   - "Talent Acquisition"
   - "Recruitment Specialist"
   - "HR Executive"
   - "Technical Recruiter"
   - "Hiring Partner"
   - "Sourcing Specialist"
   - "Executive Search Consultant"
   - "HR Manager"
   - "VP of People"
   - "Chief People Officer"
   - "CHRO"
   - And any similar role involved in hiring and recruitment.
7. If the query is ambiguous, default to the job title or keywords from the Job Description.
8. Your output must be concise and directly usable as a semantic query — no extra formatting.
*9. Review the user's conversation each time before generating the query to fully understand their requirements.
ONLY return the search phrase.
    """

    messages = [
        {"role": "system", "content": "You are a semantic search expert helping translate user queries into vector DB search queries."},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1
    )

    cleaned_query = response.choices[0].message.content.strip()

    return cleaned_query


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


def format_chroma_results(documents):
    formatted = []
    for idx, doc in enumerate(documents):
        if not doc:
            continue
        lines = doc.split("\n")
        record = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()

        formatted.append(
            f"""{idx+1}. Candidate ID: {record.get('reference', 'N/A')}
Full Name: {record.get('fullName', 'N/A')}
Gender: {record.get('gender', 'N/A')}
Job Title: {record.get('jobTitle', 'N/A')}
Email: {record.get('email', 'N/A')}
Skills: {record.get('skills', 'N/A')}
Location: {record.get('address', 'N/A')}
Nationality: {record.get('nationalities', 'N/A')}"""
        )
    return "\n\n".join(formatted)
