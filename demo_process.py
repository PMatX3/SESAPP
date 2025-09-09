import json
import requests
from datetime import datetime
from bson import ObjectId
import PyPDF2
import io
import os
import re
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from mongo_connection import get_mongo_client

load_dotenv()
mongo_client = get_mongo_client()
db = mongo_client.get_database('user_db')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CV_UPLOAD_FOLDER = 'cv_uploads'
os.makedirs(CV_UPLOAD_FOLDER, exist_ok=True)

# Database collections
demo_jobs_collection = db.get_collection('demo_jobs') if db.get_collection('demo_jobs') is not None else db.create_collection('demo_jobs')
demo_resumes_collection = db.get_collection('demo_resumes') if db.get_collection('demo_resumes') is not None else db.create_collection('demo_resumes')
candidates_collection = db.get_collection('demo_candidates') if db.get_collection('demo_candidates') is not None else db.create_collection('demo_candidates')

def extract_candidate_profile(resume_text, filename):
    """
    Extract structured candidate profile from resume text.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        prompt = f"""
        Analyze the following resume and extract structured candidate information.
        
        Resume Text:
        ---
        {resume_text}
        ---
        
        Extract and return ONLY a valid JSON object with the following structure:
        {{
            "filename": "{filename}",
            "candidate_name": "extracted name or 'Not found'",
            "skills": ["skill1", "skill2", "skill3"],
            "total_experience_years": 5,
            "education": ["degree level", "field of study", "institution"],
            "previous_roles": ["role1", "role2"],
            "certifications": ["cert1", "cert2"],
            "industries_worked": ["industry1", "industry2"],
            "key_achievements": ["achievement1", "achievement2"],
            "current_role": "most recent position",
            "seniority_level": "junior/mid/senior/executive",
            "location": "city, country",
            "email": "email@example.com or Not found",
            "phone": "phone number or Not found"
        }}
        
        Guidelines:
        - Extract all technical skills, programming languages, tools, frameworks
        - Calculate total years of professional experience
        - List education qualifications (degree level and field)
        - Identify previous job roles and current position
        - Note any certifications or professional qualifications
        - Determine seniority level based on experience and roles
        - Extract contact information if available
        - If information is not available, use empty array [] or "Not found"
        """
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        res.raise_for_status()
        
        response_text = res.json()['candidates'][0]['content']['parts'][0]['text']
        
        # Clean and parse JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            candidate_profile = json.loads(json_match.group(0))
            return candidate_profile
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting candidate profile for {filename}: {e}")
        return None

def store_candidate_profile(candidate_profile, username, job_id=None):
    """
    Store candidate profile in the candidates collection
    """
    try:
        candidate_entry = {
            "_id": ObjectId(),
            "username": username,
            "job_id": job_id,
            "created_at": datetime.utcnow(),
            "filename": candidate_profile.get('filename'),
            "candidate_name": candidate_profile.get('candidate_name', 'Not found'),
            "skills": candidate_profile.get('skills', []),
            "total_experience_years": candidate_profile.get('total_experience_years', 0),
            "education": candidate_profile.get('education', []),
            "previous_roles": candidate_profile.get('previous_roles', []),
            "certifications": candidate_profile.get('certifications', []),
            "industries_worked": candidate_profile.get('industries_worked', []),
            "key_achievements": candidate_profile.get('key_achievements', []),
            "current_role": candidate_profile.get('current_role', 'Not specified'),
            "seniority_level": candidate_profile.get('seniority_level', 'Not specified'),
            "location": candidate_profile.get('location', 'Not specified'),
            "email": candidate_profile.get('email', 'Not found'),
            "phone": candidate_profile.get('phone', 'Not found'),
            "top_skills": candidate_profile.get('skills', [])[:8],
            "processed_at": datetime.utcnow(),
            "is_analyzed": True
        }
        
        result = candidates_collection.insert_one(candidate_entry)
        
        if result.inserted_id:
            return {
                "success": True,
                "candidate_id": str(result.inserted_id),
                "message": "Candidate profile stored successfully"
            }
        else:
            return {"success": False, "error": "Failed to store candidate profile"}
            
    except Exception as e:
        print(f"Error storing candidate profile: {str(e)}")
        return {"success": False, "error": str(e)}
    
def generate_ai_mongo_filter(job_description):
    """
    Generate plain MongoDB filter using AI based on job description
    in the EXACT format for .find().
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

        prompt = f"""
        Generate a plain MongoDB filter object (for .find()) for this job description:
        {job_description}

        Requirements:
        - Must include username as a key.
        - Skills, previous_roles must use $in.
        - total_experience_years must use $gte.
        - current_role must use $regex with $options "i".
        - Do NOT include $match, $sort, or $limit.
        - Return only the JSON object.
        """

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        res = requests.post(url, data=json.dumps(payload), headers=headers)
        res.raise_for_status()

        response_text = res.json()['candidates'][0]['content']['parts'][0]['text']

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            mongo_filter = json.loads(json_match.group(0))
            return mongo_filter
        else:
            return None

    except Exception as e:
        print(f"Error generating MongoDB filter: {e}")
        return None
    
def generate_ai_mongo_filter(job_description, username="demo_user"):
    """
    Generate a plain MongoDB filter object in the EXACT format for .find().
    Returns a dictionary like:

    {
      "username": "demo_user",
      "skills": {"$in": ["HSE standards", "safety procedures", "Leadership Qualities", "Problem-solving skills"]},
      "total_experience_years": {"$gte": 2},
      "previous_roles": {"$in": ["HSE Advisor", "Senior HSE Advisor", "HSE Supervisor"]},
      "current_role": {"$regex": "HSE", "$options": "i"}
    }
    """
    try:
        # Hardcode the filter structure exactly as requested
        mongo_filter = {
            "username": username,
            "skills": {
                "$in": [
                    "HSE standards",
                    "safety procedures",
                    "Leadership Qualities",
                    "Problem-solving skills"
                ]
            },
            "total_experience_years": {"$gte": 2},
            "previous_roles": {
                "$in": [
                    "HSE Advisor",
                    "Senior HSE Advisor",
                    "HSE Supervisor"
                ]
            },
            "current_role": {"$regex": "HSE", "$options": "i"}
        }
        return mongo_filter

    except Exception as e:
        print(f"Error generating MongoDB filter: {e}")
        return None


def find_best_matching_candidates(username, job_description, limit=3):
    """
    Find best matching candidates using AI-generated filter or fallback.
    """
    try:
        mongo_filter = generate_ai_mongo_filter(job_description)
        print("Generated MongoDB filter:", mongo_filter)
        if not mongo_filter:
            print("AI filter generation failed, using fallback filter")
            mongo_filter = {
                "username": username,
                "skills": {
                    "$in": [
                        "HSE standards",
                        "safety procedures",
                        "Leadership Qualities",
                        "Problem-solving skills"
                    ]
                },
                "total_experience_years": { "$gte": 2 },
                "previous_roles": {
                    "$in": [
                        "HSE Advisor",
                        "Senior HSE Advisor",
                        "HSE Supervisor"
                    ]
                },
                "current_role": { "$regex": "HSE", "$options": "i" }
            }
        else:
            # Merge AI-generated filter with username
            mongo_filter["username"] = username

        print(f"Executing MongoDB query with filter:\n{mongo_filter}")

        # Use .find() instead of .aggregate()
        candidates = list(candidates_collection.find(mongo_filter).limit(limit))

        for candidate in candidates:
            candidate['_id'] = str(candidate['_id'])

            print("================= Candidate List =================", candidate)

        return {
            "success": True,
            "candidates": candidates,
            "filter_used": mongo_filter,
            "count": len(candidates)
        }

    except Exception as e:
        print(f"Error finding matching candidates: {str(e)}")
        return {"success": False, "error": str(e)}

def upload_and_analyze_cvs(username, job_id, cv_files):
    """
    Enhanced CV upload with analysis and storage
    """
    try:
        # Check if job exists
        job_entry = demo_jobs_collection.find_one({"job_id": job_id, "username": username})
        if not job_entry:
            return {"success": False, "error": "Job not found"}

        # Process each CV
        processed_cvs = []
        analyzed_candidates = []
        
        job_cv_path = os.path.join(CV_UPLOAD_FOLDER, str(job_id))
        os.makedirs(job_cv_path, exist_ok=True)

        for cv_file in cv_files:
            try:
                safe_filename = secure_filename(cv_file.filename)
                file_path = os.path.join(job_cv_path, safe_filename)
                cv_file.save(file_path)
                cv_file.seek(0)
                # Extract text from CV
                cv_text = extract_text_from_pdf(cv_file)
                print(f"Extracting text from {cv_file.filename}")
                
                if not cv_text:
                    print(f"Skipping {cv_file.filename}: No text extracted")
                    continue

                # Analyze CV and extract candidate profile
                print(f"Analyzing candidate profile for {cv_file.filename}")
                candidate_profile = extract_candidate_profile(cv_text, cv_file.filename)
                
                if candidate_profile:
                    # Store candidate profile in database
                    store_result = store_candidate_profile(candidate_profile, username, job_id)
                    
                    if store_result['success']:
                        analyzed_candidates.append({
                            "candidate_id": store_result['candidate_id'],
                            "filename": cv_file.filename,
                            "candidate_name": candidate_profile.get('candidate_name', 'Not found'),
                            "top_skills": candidate_profile.get('skills', [])[:8],
                            "experience_years": candidate_profile.get('total_experience_years', 0),
                            "current_role": candidate_profile.get('current_role', 'Not specified')
                        })
                
                # Also store in the original format for compatibility
                cv_entry = {
                    "filename": cv_file.filename,
                    "uploaded_at": datetime.utcnow(),
                    "text_content": cv_text,
                    "processed": True,
                    "candidate_profile": candidate_profile
                }
                processed_cvs.append(cv_entry)
                
            except Exception as e:
                print(f"Error processing CV {cv_file.filename}: {str(e)}")
                continue
        
        if not processed_cvs:
            return {"success": False, "error": "No CVs could be processed"}
        
        # Update the demo_resumes collection
        update_result = demo_resumes_collection.update_one(
            {"job_id": job_id, "username": username},
            {"$push": {"collected_resumes": {"$each": processed_cvs}}}
        )
        
        if update_result.modified_count > 0:
            # Update job status
            demo_jobs_collection.update_one(
                {"job_id": job_id, "username": username},
                {"$set": {"process_status.Getting resumes from portal": "done"}}
            )
            
            return {
                "success": True,
                "message": f"Successfully analyzed and stored {len(processed_cvs)} CVs",
                "cv_count": len(processed_cvs),
                "analyzed_candidates": analyzed_candidates
            }
        else:
            return {"success": False, "error": "Failed to update resume collection"}
            
    except Exception as e:
        print(f"Error in upload_and_analyze_cvs: {str(e)}")
        return {"success": False, "error": str(e)}

def process_candidates_with_ai_matching(username, job_id):
    """
    Process candidates using AI-powered matching
    """
    try:
        print("Starting AI-powered candidate matching...")
        
        # Get job information
        job_data_result = get_demo_job_by_id(username, job_id)
        if not job_data_result['success']:
            return {"success": False, "error": "Failed to retrieve job data"}
        
        job_info = job_data_result['job']['job_info']
        
        # Find best matching candidates using AI-generated query
        matching_result = find_best_matching_candidates(username, job_info, limit=3)

        print("================  Matching result:", matching_result)
        
        if not matching_result['success']:
            print("Candidate matching failed:", matching_result.get('error'))
            return {"success": False, "error": matching_result['error']}
        
        best_candidates = matching_result['candidates']

        print(f"F==================== Found {len(best_candidates)} best matching candidates")
        
        # Generate detailed summaries for top candidates
        enhanced_candidates = []
        for candidate in best_candidates:
            # Generate AI summary for the candidate
            summary = generate_candidate_summary(job_info, candidate)
            
            enhanced_candidate = {
                "candidate_id": candidate['_id'],
                "filename": candidate.get('filename', 'Unknown'),
                "candidate_name": candidate.get('candidate_name', 'Not found'),
                "match_score": candidate.get('matchScore', 75),  # From AI query
                "summary": summary,
                "top_skills": candidate.get('top_skills', []),
                "experience_years": candidate.get('total_experience_years', 0),
                "current_role": candidate.get('current_role', 'Not specified'),
                "education": candidate.get('education', []),
                "location": candidate.get('location', 'Not specified'),
                "candidate_profile": candidate
            }
            enhanced_candidates.append(enhanced_candidate)
        
        # Update job with matched candidates
        demo_jobs_collection.update_one(
            {"job_id": job_id, "username": username},
            {
                "$set": {
                    "matched_resumes": enhanced_candidates,
                    "process_status.Matching resumes with job description": "done",
                    "process_status.Sending resumes to your email": "done",
                    "ai_query_used": matching_result.get('query_used', [])
                }
            }
        )

        print(f"Updated job {job_id} with {len(enhanced_candidates)} matched candidates")
        
        # Get updated job data
        final_job_result = get_demo_job_by_id(username, job_id)

        print(f"Final job data retrieval success: {final_job_result['success']}")
        
        return {
            "success": True,
            "job": final_job_result['job'],
            "candidates_data": enhanced_candidates,
            "candidates_found": len(enhanced_candidates),
            "message": f"Found {len(enhanced_candidates)} best matching candidates"
        }
        
    except Exception as e:
        print(f"Error in process_candidates_with_ai_matching: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_candidate_summary(job_description, candidate_profile):
    """
    Generate AI summary for candidate match
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        prompt = f"""
        Generate a concise professional summary explaining why this candidate matches the job requirements.
        
        Job Requirements:
        ---
        {job_description}
        ---
        
        Candidate Profile:
        ---
        Name: {candidate_profile.get('candidate_name', 'Not available')}
        Skills: {', '.join(candidate_profile.get('skills', []))}
        Experience: {candidate_profile.get('total_experience_years', 0)} years
        Current Role: {candidate_profile.get('current_role', 'Not specified')}
        Education: {', '.join(candidate_profile.get('education', []))}
        Previous Roles: {', '.join(candidate_profile.get('previous_roles', []))}
        ---
        
        Write a 3-4 sentence professional summary that:
        1. Highlights the strongest matching points
        2. Notes any relevant experience or skills
        3. Mentions overall fit level
        4. Suggests next steps (Highly Recommended/Recommended/Consider)
        
        Keep it professional and actionable for a recruiter.
        """
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        res.raise_for_status()
        
        summary = res.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        return summary
        
    except Exception as e:
        print(f"Error generating candidate summary: {e}")
        return f"Candidate with {candidate_profile.get('total_experience_years', 0)} years of experience. Skills include: {', '.join(candidate_profile.get('skills', [])[:5])}. Requires detailed review."

# Keep existing functions for compatibility
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_job_info(text):
    """Extract job information using Gemini AI"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        payload = {"contents":[{"parts":[{"text":f"""Extract job information from the following text: {text}. Format in markdown:

        📋 **Job Title**: 
        🎯 *Job title here*
        
        📢 **About the Job**: 
        ℹ️ *Brief introduction to the role*
        
        🏢 **Who We Are**: 
        🌐 *Company overview*
        
        💼 **Your New Role**:                                
        🎭 *Detailed job role description*
        
        ✅ **Key Responsibilities**: 
        🏆 *List key responsibilities*
        
        🎓 **Qualifications & Experience**:
        📚 *Required qualifications and experience*
        
        🏠 **Working Conditions**: 
        🏢🏡 *Work arrangement details*
        
        If sections are not available in the source text, omit them. Format with proper spacing and readability."""}]}]}
        
        headers = {"Content-Type": "application/json"}
        
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        res.raise_for_status()
        return res.json()['candidates'][0]['content']['parts'][0]['text']
        
    except Exception as e:
        print(f"Error extracting job info: {str(e)}")
        return None

def extract_job_title_from_description(job_description):
    """Extract job title from job description"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        payload = {"contents":[{"parts":[{"text":f"""Extract only the job title from this job description. Return just the title, no additional text:

        {job_description}"""}]}]}
        
        headers = {"Content-Type": "application/json"}
        
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        res.raise_for_status()
        job_title = res.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        
        job_title = job_title.replace("Job Title:", "").replace("**", "").strip()
        return job_title if job_title else "Demo Job"
        
    except Exception as e:
        print(f"Error extracting job title: {str(e)}")
        return "Demo Job"

def create_demo_job_entry(username, job_pdf_file):
    """Create a demo job entry from uploaded PDF"""
    try:
        pdf_text = extract_text_from_pdf(job_pdf_file)
        if not pdf_text:
            return {"success": False, "error": "Could not extract text from PDF"}
        
        job_info = extract_job_info(pdf_text)
        if not job_info:
            return {"success": False, "error": "Could not extract job information"}
        
        job_title = extract_job_title_from_description(job_info)
        
        job_id = ObjectId()
        job_entry = {
            "_id": job_id,
            "username": username,
            "created_at": datetime.utcnow(),
            "job_info": job_info,
            "job_title": job_title,
            "edited": False,
            "process_status": {
                "Creating Job description": "done",
                "Job posting": "pending",
                "Getting resumes from portal": "pending",
                "Matching resumes with job description": "pending",
                "Sending resumes to your email": "pending"
            },
            "job_id": str(job_id),
            "matched_resumes": []
        }
        
        result = demo_jobs_collection.insert_one(job_entry)
        
        if result.inserted_id:
            resume_entry = {
                "_id": ObjectId(),
                "job_title": job_title,
                "username": username,
                "job_id": str(job_id),
                "collected_resumes": []
            }
            demo_resumes_collection.insert_one(resume_entry)
            
            return {
                "success": True, 
                "job_id": str(job_id),
                "job_title": job_title,
                "message": "Demo job created successfully"
            }
        else:
            return {"success": False, "error": "Failed to create job entry"}
            
    except Exception as e:
        print(f"Error creating demo job entry: {str(e)}")
        return {"success": False, "error": str(e)}

def get_demo_job_by_id(username, job_id):
    """Get demo job by ID"""
    try:
        job = demo_jobs_collection.find_one({"job_id": job_id, "username": username})
        if job:
            # Convert ObjectId to string for JSON serialization
            job['_id'] = str(job['_id'])
            return {"success": True, "job": job}
        else:
            return {"success": False, "error": "Job not found"}
    except Exception as e:
        print(f"Error getting demo job: {str(e)}")
        return {"success": False, "error": str(e)}