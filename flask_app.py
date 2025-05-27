import eventlet
eventlet.monkey_patch()
import fitz
from flask import Flask, request, render_template, flash, jsonify, session, redirect, url_for, Response, send_file
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from langchain_community.chat_models import ChatOpenAI
import os,json
from cromadbTest import load_data, execute_query, execute_query3,execute_query2, load_pdf_data, get_chat_history, load_json_data, get_chat_list, add_chat_message, clear_collection
# from test import load_data, execute_query, load_pdf_data, get_chat_history, load_json_data, get_chat_list, add_chat_message
# from test import execute_query
from utils import get_pdf_text, get_text_chunks, send_reset_password_mail, send_email, send_demo_email
import pandas as pd
import uuid
from io import BytesIO

import threading
from datetime import datetime,timedelta
import markdown
from markdown_it import MarkdownIt
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin
import jwt
import stripe
import time
from htmlTemplates import css, bot_template, user_template
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
from flask_session import Session
import gc
from mongo_connection import get_mongo_client
from cromadbTest import job_query
import logging
import html2text
import re
from flask_migrate import Migrate
from models import db  # Assuming 'db' is your SQLAlchemy instance
import time

app = Flask(__name__)


# Initialize SQLAlchemy
# Import the models
from models import BookingAppointment


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/ubuntu/SES_Flask/bookings_storage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
db.init_app(app)
# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

migrate = Migrate(app, db)

CORS(app, supports_credentials=True)
socketio = SocketIO(app, ping_timeout=240, ping_interval=120)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Adjust based on your requirements

cancellation_flag = threading.Event()

socketio.init_app(app, cors_allowed_origins="*")

stripe_keys = {
    "secret_key": os.environ["STRIPE_SECRET_KEY"],
    "publishable_key": os.environ["STRIPE_PUBLISHABLE_KEY"],
}

stripe.api_key = stripe_keys["secret_key"]

client = get_mongo_client()
mongo = client['user_db']

app.secret_key = '31dee9b85d3be7513eda6e3bb1b2e22edd923194d18b3cf8'

UPLOAD_FOLDER = 'uploads'
JOB_DESCRIPTION_FOLDER = 'job_descriptions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



DOMAIN = 'https://www.yourbestcandidate.ai'

md = (
    MarkdownIt('commonmark' ,{'breaks':True,'html':True})
    .use(front_matter_plugin)
    .use(footnote_plugin)
    .enable('table')
)


import logging

# Setup logging with a file name
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_cache_json_data(user_id, filename=None, temp=False):

    result_holder = {}

    def load_json():
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            update_user_file_reference(user_id, 'data_file', csv_filename)
            result = load_json_data(csv_filename, data, temp=temp)
            result_holder.update(result)

    thread = threading.Thread(target=load_json)
    thread.start()
    thread.join()

    return result_holder

# def load_and_cache_json_data(filename=None, user_id=None, temp=False):
#     def load_json():
#         if filename and temp:
#             with open(filename, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#                 load_json_data(data, user_id,True)
#         else:
#             with open(filename, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#                 load_json_data(data, user_id)
        

#     # Start a new thread for loading JSON data
#     thread = threading.Thread(target=load_json)
#     thread.start()

def load_and_cache_file_data(filename=None, temp=False):
    def load_file():
        load_data(filename, temp)
        

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_file)
    thread.start()

@app.route('/files', methods=['GET'])
def list_files():
    company_name = request.args.get('companyName')
    root_dir = 'company_data'
    allowed_extensions = {'.pdf', '.csv', '.json'}
    files_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if company_name in filename and os.path.splitext(filename)[1] in allowed_extensions:
                files_list.append(filename)

    return jsonify(files_list)


# @app.route('/robots.txt', methods=['GET'])
# def list_files():
#     return 'robots.txt'

def check_for_file(company_name):
    root_dir = 'company_data'  # Start from the root of the filesystem
    file_found = None
    allowed_extensions = {'.pdf', '.csv', '.json'}  # Set of allowed file extensions

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            print(filename)
            if company_name in filename and os.path.splitext(filename)[1] in allowed_extensions:
                file_found = os.path.join(dirpath, filename)
                break
        if file_found:
            print("file_found => ",file_found)
            return file_found
    else:
        return None

def calculate_days_since_join(join_date_str):
    join_date = datetime.strptime(join_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    difference = current_date - join_date
    return difference.days
@app.route('/')
def index():
    return render_template('hompage2.html')

@app.route('/chat')
def chat():
    job_desc = job_query.get('job_profile')
    if job_desc['documents']:
        job_desc['documents'] = []
    if 'user_id' in session:
        user_id = session['user_id']
        try:
            session.pop('current_data_source', None)
            session.pop('current_file_path',None)
            session.pop('job_description_text',None)
            session.pop('pdf_file_path',None)
            clear_collection(user_id)
        except Exception as e:
            print("error in clear collection => ",e)
        user_data = mongo.users.find_one({'app_name': user_id})
        if user_data:
            days_since_join = calculate_days_since_join(user_data['join_date'])
            days = user_data.get('days', 0)
            company = user_data.get('company', '')
            login_attempts = user_data.get('login_attempts')
            session['days_since_join'] = days_since_join
            session['days'] = days
            session['company'] = company
            session['login_attempt'] = login_attempts
            print("company => ",company)
            if company == 'SES':
                print("in if")
                file_path = check_for_file(company)
                if file_path:
                    load_and_cache_json_data(file_path, True)
            # if days_since_join >= days:
            #     return redirect(url_for('pricing'))
            # else:
            return render_template('index3.html')
        else:
            return redirect('/')
    else:
        return redirect('/')
    
#PAYMENT PART    
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/terms_and_conditions')
def terms_and_conditions():
    return render_template('terms_of_services.html')

@app.route('/privacy_policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route("/charge", methods=["POST"])
def charge():
    return render_template('coming_soon.html')
    # amount = 1000  # amount in cents
    # currency = "usd"
    # description = "A Flask Charge"

    # stripe.api_key = stripe_keys["secret_key"]
    # print("stripe_keys ==> ",stripe_keys)
    # create_checkout_session = stripe.checkout.Session.create(
    #     line_items=[
    #         {
    #             'price':'price_1P9ng8DJlqPYNFXOetpsVLtL',
    #             'quantity':1
    #         }
    #     ],
    #     mode='subscription',
    #     success_url=f'{DOMAIN}/',
    #     cancel_url=f'{DOMAIN}/pricing'
    # )
    # print("123 ---- ==== ")
    # print("create_checkout_session['payment_status'] => ", create_checkout_session['payment_status'])
    # if create_checkout_session['payment_status'] == 'unpaid':
    #     with app.test_client() as client:
    #         print("in with")
    #         print("session['user_id'] => ", session['user_id'])
    #         response = client.post('/edit/' + session['user_id'], data={'new_days': 100, 'new_join_date': datetime.now().date().isoformat()})
    #         print("response = > ", response)
    #         if response.status_code == 200:
    #             print('Days updated to 100 for unpaid session.')
    #         else:
    #             print('Failed to update days for unpaid session.')
    # print("12333")
    # print("check out url", create_checkout_session.url)
    # return redirect(create_checkout_session.url,303)

# admin part
@app.route('/admin')
def admin():
    if 'user_id' in session and 'user_type' in session:
        print(session['user_type'])
        if session['user_type'] != 'admin':
            return redirect('/')
        current_user_id = session['user_id']
        users = mongo.users.find()
        user_list = [user for user in users if user['app_name'] != current_user_id]  

        return render_template('admin.html', users=user_list)
    else:
        print("Redirecting to login from admin due to missing user_id in session")
        return redirect(url_for('login'))

@app.route('/insert_user', methods=['GET','POST'])
def insert_user():
    if request.method == 'POST':
        try:
            user_data = request.form
            app_name = user_data.get('app_name')
            password = user_data.get('password')
            login_attempts = user_data.get('login_attempts')
            days = user_data.get('days')
            user_type = user_data.get('user_type')
            first_name = user_data.get('first_name')
            last_name = user_data.get('last_name')
            company = user_data.get('company')

            if not app_name or not password or not first_name or not last_name:
                return jsonify({'message': 'Please fill out all required fields.'}), 500
            
            # Check if user already exists
            existing_user = mongo.users.find_one({'app_name': app_name})
            if existing_user:
                return jsonify({'message': 'Email already registered.'}), 500

            # Get the current date and convert to string
            join_date = datetime.now().date().isoformat()

            # Insert the new user into the MongoDB
            mongo.users.insert_one({
                'app_name': app_name,
                'password': password,
                'login_attempts': int(login_attempts),
                'days': int(days),
                'company': company,
                'type': user_type,
                'join_date': join_date,
                'first_name': first_name,
                'last_name': last_name
            })
            return jsonify({'success':True, 'message': 'Data inserted successfully!'}), 200

        except Exception as e:
            return jsonify({'success':False, 'message': str(e)}), 500
    else:
        return render_template('insert.html')

@app.route('/edit/<app_name>', methods=['GET', 'POST'])
def edit(app_name):
    if request.method == 'POST':
        new_data = request.form.get('new_data')
        new_password = request.form.get('new_password')
        new_days = request.form.get('new_days', type=int)
        new_login_attempts = request.form.get('new_login_attempts', type=int)
        new_company = request.form.get('new_company')
        new_usertype = request.form.get('user_type')
        new_join_date = request.form.get('new_join_date', None)
        new_first_name = request.form.get('new_first_name')
        new_last_name = request.form.get('new_last_name')

        update_fields = {
            'app_name': new_data,
            'password': new_password,
            'days': new_days,
            'login_attempts': new_login_attempts,
            'company': new_company,
            'type': new_usertype,
            'join_date': new_join_date,
            'first_name': new_first_name,
            'last_name': new_last_name
        }

        # Remove keys where values are None to avoid overwriting with None
        update_fields = {k: v for k, v in update_fields.items() if v is not None}

        mongo.users.update_one({'app_name': app_name}, {'$set': update_fields})
        return jsonify({'success':True, 'message': 'Data updated successfully!'}), 200
    else:
        app_data = mongo.users.find_one({'app_name': app_name})
        if app_data:
            return render_template('edit.html', app_name=app_name, app_data=app_data)
        else:
            return jsonify({'error': 'No data found for the specified app name.'}), 404

@app.route('/delete/<app_name>', methods=['GET','POST'])
def delete(app_name):
    # Logic to delete the record associated with app_name
    delete_result = mongo.users.delete_one({'app_name': app_name})
    mongo.chat_history.delete_many({'user_id': app_name})
    if delete_result.deleted_count > 0:
        flash('Record deleted successfully!')
    else:
        flash('No record found with the specified app name.')
    return redirect(url_for('admin'))

@app.route('/updated_cmp', methods=['POST'])
def updated_cmp():
    data = request.json
    app_name = data.get('app_name')
    ses = data.get('comapany')
    
    if not app_name or ses is None:
        return jsonify({'error': 'Missing app_name or ses'}), 400

    update_result = mongo.users.update_one(
        {'app_name': app_name},
        {'$set': {'comapany': ses}}
    )

    if update_result.modified_count > 0:
        return jsonify({'message': 'SES updated successfully'}), 200
    else:
        return jsonify({'error': 'No record found with the specified app_name or update failed'}), 404

# user part
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if 'user_id' in session:
            return redirect('/')

        # Extract data from form
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        company = request.form.get('company')
        email = request.form.get('email')
        password = request.form.get('password')

        # Basic validation
        if not first_name or not last_name or not email or not password:
            return jsonify({'message': 'Please fill out all required fields.'}), 500

        # Check if user already exists
        existing_user = mongo.users.find_one({'app_name': email})
        if existing_user:
            return jsonify({'message': 'Email already registered.'}), 500

        # Get the current date and convert to string
        join_date = datetime.now().date().isoformat()

        # Insert new user into the database with default values for new fields
        mongo.users.insert_one({
            'first_name': first_name,
            'last_name': last_name,
            'company': company,
            'app_name': email,
            'password': password,
            'days': 14,  
            'login_attempts': 0,  
            'type': 'user',  
            'join_date': join_date  
        })

        return redirect(url_for('login'))
    else:
        return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with app.app_context():
            app_name = request.form.get('app_name')
            password = request.form.get('password')
            with app.test_client() as client:
                response = client.get(f'/get_password?app_name={app_name}')
            if response.status_code == 200:
                data = response.json
                password_from_api = data['password']
                login_attempt = data['login_attempts']
                days = data['days']
                join_date = data['join_date']
                user_type = data['type']
                company = data['company']
                days_since_join = calculate_days_since_join(join_date)
                print('days since ---- >>>> ',days_since_join)
            else:
                password_from_api = None
                login_attempt = None
                days = None
                user_type = None
                days_since_join = 0
                company = None

            session['days_since_join'] = days_since_join
            session['days'] = days
            session['company'] = company
            session['user_type'] = user_type
            session['login_attempt'] = login_attempt
            if login_attempt == 0:
                session['user_id'] = app_name
                return jsonify({'login_attempt':login_attempt,'days_since_join':days_since_join})
            
            if password_from_api:
                if password_from_api == password:
                    # Generate a new chat_id using uuid
                    chat_id = str(uuid.uuid4())
                    session['user_id'] = app_name
                    session['chat_id'] = chat_id  
                    with app.test_client() as client:
                        client.post(url_for('update_login_attempts'), data={'app_name': app_name})
                    if user_type == 'admin':
                        return jsonify({'user':'admin'})
                    else:
                        return jsonify({'user':'user'})
                else:
                    return jsonify({'message': 'Invalid credentials'}), 401
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
    elif request.method == 'GET':
        if 'user_id' not in session:
            session.clear()
            return render_template('login.html')
        else:
            return redirect('/chat')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    # if user_id:
    #     for filename in os.listdir(app.config['UPLOAD_FOLDER']):
    #         if user_id in filename:
    #             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #             try:
    #                 os.remove(file_path)
    #                 print(f"Deleted {file_path}")
    #             except FileNotFoundError:
    #                 print(f"File not found: {file_path}")
    # Clear the session
    session.clear()
    # Redirect to login page
    return redirect(url_for('login'))

@app.route('/chats', methods=['GET', 'POST'])
def chats():
    with app.app_context():
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login'))

        chat_list = get_chat_list(user_id)
        chat_history = None
        if request.method == 'POST':
            data = request.get_json()  # Get JSON data from request
            selected_chat_id = data.get('chat_id')  # Extract chat_id from JSON data
            print("selected_chat_id => ", selected_chat_id)
            if selected_chat_id:
                session['chat_id'] = selected_chat_id
                session['user_id'] = user_id
                
                # Use test_client to make a request to the local API
                with app.test_client() as client:
                    # Ensure the session is passed to the test client
                    with client.session_transaction() as sess:
                        sess['user_id'] = user_id
                        sess['chat_id'] = selected_chat_id
                    
                    # Make a GET request to the local API endpoint
                    response = client.get('/get_chat_history')
                if response.status_code == 200:
                    chat_history = response.json
                else:
                    chat_history = {'error': 'Failed to retrieve chat history'}
                return jsonify({'chat_list': chat_list, 'chat_history': chat_history, 'selected_chat_id': selected_chat_id})
        
        return jsonify({'chat_list': chat_list, 'chat_history': chat_history})

@app.route('/new_chat')
def new_chat():
    try:
        if 'user_id' in session:
            user_id = session.get('user_id')
            try:
                session.pop('current_data_source', None)
                session.pop('current_file_path',None)
                session.pop('job_description_text',None)
                session.pop('pdf_file_path',None)
                clear_collection(user_id)
            except Exception as e:
                print("error in clear collection => ",e)
            user_conversations[user_id] = []
            # Generate a new chat_id using uuid
            chat_id = str(uuid.uuid4()) 
            # Set the new chat_id in the session
            session['chat_id'] = chat_id
            print("chat_id from new_chat ===== ", chat_id)
            # response.headers['Location'] += '?new_chat=false'  # Optionally set new_chat to false
            return render_template('index3.html')
            # response = redirect(url_for('index'))  # Redirect to index without new_chat in URL
            return response
        else:
            return redirect('/login')
    except Exception as e:
        print("=============")
        print(e)
        print("=============")
        return redirect(url_for('login'))

@app.route('/change_password')
def change_password():
    token = request.args.get('token', None)
    if token:
        try:
            decoded_token = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            user_id = decoded_token.get('user_id')
            user = mongo.users.find_one({'app_name': user_id})
            if user:
                join_date = user['join_date'] 
                days = user['days'] 
                company = user['company'] 
                days_since_join = calculate_days_since_join(join_date)
                session['days_since_join'] = days_since_join
                session['days'] = days
                session['company'] = company
            return render_template('change_password.html', user_id=user_id)
        except jwt.ExpiredSignatureError:
            return 'Token has expired', 401
        except jwt.InvalidTokenError:
            return 'Invalid token', 401
    user_id = session.get('user_id')
    return render_template('change_password.html', user_id=user_id)

@app.route('/reset_password', methods=['GET','POST'])
def reset_password():
    if request.method == 'POST':
        user_data = request.form
        user_id = user_data.get('userEmail')
        user = mongo.users.find_one({'app_name': user_id})
        firstname = user['first_name']

        # Generate JWT token
        exp = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
        reset_token = jwt.encode({'user_id': user_id, 'exp': exp}, app.secret_key, algorithm='HS256')
        path = url_for('change_password', token=reset_token)
        reset_link = f"{DOMAIN}{path}"
        # send_reset_password_mail(user_id,'Reset Password link', firstname, reset_link)
        send_email([user_id],'Reset Password link', firstname, reset_link)
        return jsonify({'message':'email sent!'})
    else:
        return render_template('reset_password.html')


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

# @app.route('/load_data', methods=['POST'])
# def load_new_data():
#     print(" Load data clled \n")
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'User ID not found in session'}), 401
    
#     files = request.files.getlist('file')
#     use_ses = request.form.get('useSES') == 'true'
#     job_description_folder = 'job_descriptions'
    
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)

#     if not os.path.exists(job_description_folder):
#         os.makedirs(job_description_folder)
    
#     if files and files[0].filename:
#         for file in files:
#             filename = secure_filename(file.filename)
#             _, file_ext = os.path.splitext(filename)
#             unique_id = f"{uuid.uuid4()}-{user_id}"
#             filepath = os.path.join(UPLOAD_FOLDER, f"{unique_id}{file_ext}")

#             try:
#                 file.save(filepath)
#             except Exception as e:
#                 return jsonify({'error': f'Error saving file {filename}: {e}'}), 500    

    
#             if file_ext.lower() == 'pdf':
#                 pdf_file_path = os.path.join(job_description_folder,  secure_filename(file.filename))
#                 file.save(pdf_file_path)  # Save the PDF to the upload folder
#                 session['pdf_file_path'] = pdf_file_path  # Store the path in session
#                 job_description = extract_text_from_pdf(pdf_file_path)
#                 session['job_description'] = job_description  # Save the extracted job description
#                 message=  "PDF uploaded and processed successfully!"

#             if use_ses:
#                 session['ses_data'] = True
#                 session['current_data_source'] = 'ses'
#                 session.pop('current_file_path', None)
#                 return jsonify({"message": "SES data loaded successfully!"}), 200

#             if not file:
#                 return jsonify({"message": "No file uploaded"}), 400
            
#             if file_ext.lower() == 'csv':
#                 print(" inside csv upload  \n")
#                 csv_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#                 file.save(csv_path)
#                 load_and_cache_file_data(csv_path)
#                 session['current_file_path'] = csv_path
#                 session['current_data_source'] = 'csv'
#                 session.pop('ses_data', None)
#                 return jsonify({"message": "CSV data loaded successfully!"}), 200

#             if file_ext.lower() == 'json':
#                 json_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#                 file.save(json_path)
#                 load_and_cache_json_data(json_path, True)
#                 json_converted_path = 'csvdata/json_data.csv'
#                 session['current_file_path'] = json_converted_path
#                 session['current_data_source'] = 'json_data'
#                 session.pop('ses_data', None)
#                 return jsonify({"message": "JSON data loaded successfully!"}), 200   
            
#     print(" ======= Session ======== ", session)        
#     return jsonify({"message": "Invalid file type provided."}), 400



#AI Part
@app.route('/load_data', methods=['POST'])
def load_new_data():
    print("============ Load data api called =============")
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID not found in session'}), 401

    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_FOLDER, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    clear_collection(user_id)
    
    message = ''
    errors = []
    start_time = time.time()

    files = request.files.getlist('file')
    use_ses_from_form = request.form.get('useSES')
    
    if use_ses_from_form == 'true':
        session['useSES'] = True
        session['current_file_path'] = None
    else:
        message = "No files uploaded and not using SES data."
        session['useSES'] = False



    try:
        if files and files[0].filename:
            for file in files:
                filename = secure_filename(file.filename)
                _, file_ext = os.path.splitext(filename)
                unique_filename  = f"{uuid.uuid4()}-{file_ext}"
                filepath = os.path.join(user_dir, unique_filename)

                try:
                    file.save(filepath)
                except Exception as e:
                    return jsonify({'error': f'Error saving file {filename}: {e}'}), 500

                # Handle PDF
                if file_ext.lower() == '.pdf':
                    print("=============== New PDF Uploaded =========")

                    # Clear previous PDF data from session

                    # Extract text from the new PDF
                    raw_text = extract_text_from_pdf(filepath)

                    if raw_text.startswith("Error:"):
                        os.remove(filepath)
                        return jsonify({'error': f'Error processing PDF: {raw_text}'}), 500

                    # Generate new .txt filename and path
                    txt_filename = f"jobdescription_{user_id}.txt"
                    txt_path = os.path.join(JOB_DESCRIPTION_FOLDER, txt_filename)
                    update_pdf_file_reference(user_id, 'job_description', txt_path)

                    # Save extracted text to the .txt file
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(raw_text)

                    # Update session with the new file path and text
                    session['job_description_text'] = raw_text

                    message = "PDF successfully processed and converted to text"

                elif file_ext.lower() == '.csv' :
                    try:
                        load_and_cache_file_data( filepath)
                        update_user_file_reference(user_id, 'data_file', filepath)
                        session.pop('useSES', None)
                        
                        message = "CSV file successfully processed"
                    except Exception as e:
                        os.remove(filepath)
                        return jsonify({'error': f'Error processing CSV: {e}'}), 500

                elif file_ext.lower() == '.json':
                    try:
                        load_and_cache_json_data(user_id, filepath, True)
                        session.pop('useSES', None)
                        
                        message = "JSON file successfully processed"
                    except Exception as e:
                        os.remove(filepath)
                        return jsonify({'error': f'Error processing JSON: {e}'}), 500

                else:
                    os.remove(filepath)
                    return jsonify({'error': f'Invalid or duplicate file uploaded: {filename}'}), 400
                
        elapsed = round(time.time() - start_time, 2)
        session.modified = True

        print(" Session ---- ", session )
        print(" load_time_seconds ---- ", elapsed )

        return jsonify({
            'message': message,
            'errors': errors,
            'useSES': session.get('useSES', False),
            'load_time_seconds': elapsed
        }), 200

    except Exception as e:
        print(" Exception ---- ", e )
        return jsonify({'error': f'Unexpected error: {e}'}), 500

def update_pdf_file_reference(user_id,file_type , file_path):
        mongo.users.update_one(
        {'app_name': user_id},
        {'$set': {file_type: file_path}},
        upsert=True
    )

def update_user_file_reference(user_id, file_type, file_path):
    
    # Using MongoDB as an example - adjust based on your database
    mongo.users.update_one(
        {'app_name': user_id},
        {'$set': {file_type: file_path}},
        upsert=True
    )


    # if 'json' in data_types:
    #     user_id = session.get('user_id')
    #     session['recrutly_id'] = True
    #     user = mongo.users.find_one({'app_name': user_id})
    #     if user:
    #         company_name = user.get('company', None)
        
    #     if company_name:

    #         file_path = check_for_file(company_name) if company_name not in ['', None] else None
    #         if file_path:
    #                 load_and_cache_json_data(file_path, True)
    #                 session['current_data_source'] = 'json_existing'
    #                 session['current_file_path'] = file_path  # Store path for existing data too
    #                 if message:
    #                     message += f' and {company_name} data processed.'
    #                 else:
    #                     message = f'{company_name} data processed.'
    #         else:
    #             return jsonify({'error': f"No data found for company: {company_name}"}), 404 

    #         if message is not None:
    #             message += ' and '+f'{company_name} data processed.'
    #         else:
    #             message = f'{company_name} data processed.'
    # print(" ============== Session : ", session)          
    # return jsonify({'message': message, 'current_data_source': session.get('current_data_source'), 'current_file_path': session.get('current_file_path')}), 200

# Flask route to handle CSV download
@app.route('/download_csv/<filename>', methods=['GET'])
def download_csv(filename):
    file_path = f"tmp/{filename}"
    return send_file(file_path, as_attachment=True)

# Create a global dictionary to store user conversations
user_conversations = {}

@socketio.on('ask')
def handle_ask(json):
    # print("handle_ask called with data:", json)
    try:
        global cancellation_flag
        cancellation_flag.clear()
        user_question = json['question']
        user_id = json.get('user_id')
        chat_id = json.get('chat_id')
        current_data_source = json.get('dataSource')

        print("\n ============================= Current Data source =========================\n", current_data_source)
        
        recruitly_data = json.get('recruitly_data', False)
        message_id = str(uuid.uuid4())
        print('recruitly_data => ', recruitly_data)
        try:
            # Use test_client to make a request to the local API
            with app.test_client() as client:
                # Ensure the session is passed to the test client
                with client.session_transaction() as sess:
                    sess['chat_id'] = str(chat_id)  # Use session variable
                    sess['user_id'] = str(user_id)  # Use session variable

                # Make a GET request to the local API endpoint
                print("Making GET request to /get_chat_history")
                response = client.get('/get_chat_history')
                # print("Received response for chat history:")
                chat_history = response.json
        except Exception as e:
            print("Error fetching chat history:", e)
            chat_history = []

        if not user_id or not chat_id:
            error_message = 'User ID or Chat ID missing from session'
            print("Error:", error_message)
            emit('error', {'error': error_message})
            return

        # Initialize conversation history for the user if not already present
        if user_id not in user_conversations:
            user_conversations[user_id] = []
            print(f"Initialized conversation history for user: {user_id}")

        entire_response = ''
        continuation_token = None
        retry = False
        text_content = ''

        # Append the current question to the user's conversation history
        user_conversations[user_id].append({'question': user_question})
        # print(f"Appended user question to history for user {user_id}: {user_question}")

        # remove Html, \\ , and links from the conversation
        if 'error' in chat_history:
            print("Chat history contains an error, clearing it.")
            chat_history = []

        for chat in chat_history:
            # print("Processing chat history item:", chat)
            if 'ai' in chat:
                chat['ai'] = html2text.html2text(chat['ai'])
                chat['ai'] = chat['ai'].replace('\n', '').replace('*', '')
                chat['ai'] = re.sub(r'\!\[.*?\]\(.*?\)', '', chat['ai'])
                chat['ai'] = chat['ai'].replace('\\', '')
                
            if 'user' in chat:
                chat['user'] = html2text.html2text(chat['user'])
                chat['user'] = chat['user'].replace('\n', '').replace('*', '')
                chat['user'] = re.sub(r'\!\[.*?\]\(.*?\)', '', chat['user'])
                chat['user'] = chat['user'].replace('\\', '')
                

        while True:
            finish_res = None  # Initialize finish_res at the start of the loop
            print("Entering query execution loop.")
            for chunk, temp_finish_res in execute_query3(user_question, user_id, continuation_token, user_conversation=user_conversations, current_data_source = current_data_source):
                if cancellation_flag.is_set():
                    print("Cancellation flag is set, breaking the inner loop.")
                    break
                if retry:
                    print("Retry is True, removing last 400 characters from entire_response.")
                    entire_response = entire_response[:-400]
                    retry = False
                entire_response += chunk
                rendered_response = md.render(entire_response)
                # print("Rendered response:", rendered_response)

                # Store the response in the user's conversation history
                text_content = html2text.html2text(rendered_response)
                text_content = text_content.replace('\n', '').replace('*', '')

                # Spawn a new thread to handle message saving
                
                threading.Thread(target=add_chat_message, args=(user_id, user_question, rendered_response, chat_id, message_id)).start()
                # print(f"----------------------- \n\n ------------------------- {rendered_response} \n\n ------------------------- \n\n -----------------------")
                emit('message', {'data': rendered_response, 'is_complete': temp_finish_res})
                # print("Emitted message to the client.")

                # Clear the chunk to free memory
                del chunk
                # Force garbage collection to free memory
                gc.collect()
                finish_res = temp_finish_res  # Update finish_res from the loop variable
            user_conversations[user_id][-1]['response'] = text_content
            print(" ================ temp response : ",temp_finish_res )
            # print(f"Updated user conversation history with response for user {user_id}.")
            if finish_res == 'length':
                retry = True
                continuation_token = f'Your response : {rendered_response} got cut off, because you only have limited response space. Continue writing exactly where you left off based on context : Context. Do not repeat yourself. Start your response exact with: "{entire_response[-400:]}", don\'t forget to respect format of response based on previous response and don\'t start with like "Here is response" or anything just start from where you left'
                print("Response length limit reached, setting up for continuation.")
                continue  # Continue the loop to process the next part of the query
            # elif finish_res == 'csv_download':
            #     print('done hrer')
            #     download_link = f"https://yourbestcandidate.ai/download_csv/{user_id}_results.csv"
            #     entire_response += f"We have large data and because of token limitation I am not able to respond, but here is the download button of that data in CSV: [Download CSV]({download_link})"
            #     rendered_response = md.render(entire_response)
            #     emit('message', {'data': rendered_response, 'is_complete': 'stop'})
            #     break
            else:
                print("Query execution finished with status:", finish_res)
                
                break

        # Clear entire_response if no longer needed
        del entire_response
        gc.collect()
        socketio.emit('task_cancelled', {'data': 'Task cancellation requested'})
        print("Cleaned up entire_response and triggered garbage collection.")

    except Exception as e:
        error_message = str(e)
        print("Exception in handle_ask:", error_message)
        emit('error', {'error': error_message})

@socketio.on('ask2')
def handle_ask2(json):
    # print("handle_ask2 called with data:", json)
    try:
        global cancellation_flag
        cancellation_flag.clear()
        user_question = json['question']
        user_id = json.get('user_id')
        chat_id = json.get('chat_id')
        recruitly_data = json.get('recruitly_data', False)

        try:
            # Use test_client to make a request to the local API
            with app.test_client() as client:
                # Ensure the session is passed to the test client
                with client.session_transaction() as sess:
                    sess['chat_id'] = session['chat_id']  # Use session variable
                    sess['user_id'] = session['user_id']  # Use session variable

                # Make a GET request to the local API endpoint
                print("Making GET request to /get_chat_history")
                response = client.get('/get_chat_history')
                print("Chat history response ===== \n\n", response.json, "\n\n =====")
                chat_history = response.json
        except Exception as e:
            print("Error fetching chat history in handle_ask2:", e)
            chat_history = []

        message_id = str(uuid.uuid4())
        print("Question received in handle_ask2 ===> ", user_question)
        if not user_id or not chat_id:
            error_message = 'User ID or Chat ID missing from session'
            print("Error in handle_ask2:", error_message)
            emit('error', {'error': error_message})
            return
        entire_response = ''
        continuation_token = None
        retry = False

        while True:
            finish_res = None  # Initialize finish_res at the start of the loop
            print("Entering query execution loop in handle_ask2.")
            for chunk, temp_finish_res in execute_query2(user_question, user_id, recruitly_data, continuation_token):
                
                if temp_finish_res == 'csv':
                    print("Received CSV data, emitting message.")
                    
                    emit('message', {'csv_data': chunk, 'is_complete': 'csv'})
                    print("Spawning thread to save CSV chat message.")
                    threading.Thread(target=add_chat_message, args=(user_id, user_question, chunk, chat_id, message_id)).start()
                    break
                if cancellation_flag.is_set():
                    print("Cancellation flag is set, breaking the inner loop in handle_ask2.")
                    break
                if retry:
                    print("Retry is True, removing last 400 characters from entire_response in handle_ask2.")
                    entire_response = entire_response[:-400]
                    retry = False
                entire_response += chunk
                rendered_response = md.render(entire_response)
                # Spawn a new thread to handle message saving
                print("Spawning thread to save chat message in handle_ask2.")
                logging.INFO(f"response data : {rendered_response}")
                threading.Thread(target=add_chat_message, args=(user_id, user_question, rendered_response, chat_id, message_id)).start()
                emit('message', {'data': rendered_response, 'is_complete': temp_finish_res})
                print("Emitted message to the client in handle_ask2.")

                # Clear the chunk to free memory
                del chunk
                # Force garbage collection to free memory
                gc.collect()
                finish_res = temp_finish_res  # Update finish_res from the loop variable

            if finish_res == 'length':
                retry = True
                continuation_token = f'Your response : {rendered_response} got cut off, because you only have limited response space. Continue writing exactly where you left off based on context : Context. Do not repeat yourself. Start your response exact with: "{entire_response[-400:]}", don\'t forgot to respect format of response based on previous response and don\'t start with like "Here is response" or anything just start from where you left'  # Update the continuation token for the next iteration
                print("Response length limit reached in handle_ask2, setting up for continuation.")
                continue  # Continue the loop to process the next part of the query
            else:
                print("Query execution finished in handle_ask2 with status:", finish_res)
                break

        # Clear entire_response if no longer needed
        del entire_response
        gc.collect()
        print("Cleaned up entire_response and triggered garbage collection in handle_ask2.")

    except Exception as e:
        error_message = str(e)
        print("Exception in handle_ask2:", error_message)
        emit('error', {'error': error_message})

@socketio.on('cancel_task')
def handle_cancel_task():
    print("task cancel ===== > ")
    global cancellation_flag
    cancellation_flag.set()  # Set the flag to signal cancellation
    emit('task_cancelled', {'data': 'Task cancellation requested'})


@app.route('/get_chat_history', methods=['GET'])
def api_get_chat_history():
    with app.app_context():
        user_id = session.get('user_id')
        chat_id = session.get('chat_id')
    
        if not user_id or not chat_id:
            return jsonify([]), 200  # Return empty list instead of error
    
        chat_history = get_chat_history(user_id, chat_id)
        
        if chat_history:
            for message in chat_history:
                message['_id'] = str(message['_id'])  # Convert ObjectId to string
            
            formatted_history = [
                {
                    'user': user_template.replace('{{MSG}}', message['message']),
                    'ai': bot_template.replace('{{MSG}}', message['response'] if isinstance(message['response'], str) else "CSV preview is not displayed in the chat history.")
                } 
                for message in chat_history
            ]
            return jsonify(formatted_history), 200
        else:
            return jsonify([]), 200  # Also return empty array here


@app.route('/delete_chat/<chat_id>', methods=['DELETE','POST'])
def delete_chat(chat_id):
    if not chat_id:
        return jsonify({'error': 'Chat ID is required'}), 400

    delete_result = mongo.chat_history.delete_many({'chat_id': chat_id})
    if delete_result.deleted_count > 0:
        return jsonify({'message': 'Chat deleted successfully', 'success':True}), 200
    else:
        return jsonify({'error': 'No chat found with the specified chat ID', 'success':False}), 404

@app.route('/set_recruitly_data', methods=['POST'])
def set_recruitly_data():
    session['recruitly-data'] = request.json.get('recruitlyData', False)
    return jsonify(success=True)

@app.route('/store_password', methods=['POST'])
def store_password():
    app_name = request.form.get('app_name')
    password = request.form.get('password')
    user_type = request.form.get('user_type', 'user')  # Default to 'user' if not specified
    join_date = datetime.now().date().isoformat()  # Get the current date and convert to string

    mongo.users.insert_one({
        'app_name': app_name,
        'password': password,
        'login_attempts': 0,
        'days': 30,
        'type': user_type,  
        'join_date': join_date  
    })
    return 'Password stored successfully'

@app.route('/get_password', methods=['GET'])
def get_password():
    app_name = request.args.get('app_name')
    stored_password = mongo.users.find_one({'app_name': app_name})
    if stored_password:
        stored_password['_id'] = str(stored_password['_id'])  # Convert ObjectId to string
        return jsonify(stored_password), 200
    else:
        return jsonify({'error': 'Password not found'}), 404

@app.route('/update_password', methods=['POST'])
def update_password():
    app_name = request.form.get('app_name')
    new_password = request.form.get('new_password')
    password_record = mongo.users.find_one({'app_name': app_name})
    if password_record:
        session['user_id'] = app_name
        mongo.users.update_one(
            {'app_name': app_name},
            {'$set': {'password': new_password}}
        )
        with app.test_client() as client:
            client.post(url_for('update_login_attempts'), data={'app_name': app_name})
        return 'Password updated successfully'
    else:
        return 'Password record not found'

@app.route('/update_login_attempts', methods=['POST'])
def update_login_attempts():
    app_name = request.form.get('app_name')
    increment = request.form.get('increment', type=int, default=1)
    password_record = mongo.users.find_one({'app_name': app_name})
    if password_record:
        new_login_attempts = password_record.get('login_attempts', 0) + increment
        mongo.users.update_one(
            {'app_name': app_name},
            {'$set': {'login_attempts': new_login_attempts}}
        )
        return 'Login attempts updated successfully'
    else:
        return 'Password record not found'

@app.route('/get_all_credentials', methods=['GET'])
def get_all_credentials():
    passwords = mongo.users.find()
    credentials = []
    for password in passwords:
        password['_id'] = str(password['_id'])  # Convert ObjectId to string
        credentials.append(password)
    return jsonify(credentials)

@app.route('/check_slot_is_booked', methods=['GET'])
def check_slot_is_booked():
    date = request.args.get('date')
    timeslot = request.args.get('timeslot')
    existing_booking = BookingAppointment.query.filter_by(date=date, timeslot=timeslot).first()
    if existing_booking:
        return jsonify({"is_booked": True}), 200
    else:
        return jsonify({"is_booked": False}), 200

@app.route('/book_demo', methods=['POST'])
def book_demo():
    # Get the data from the request
    name = request.json.get('name')
    email = request.json.get('email')
    phone = request.json.get('phone')
    date = request.json.get('date')
    timeslot = request.json.get('timeslot')
    message = request.json.get('message')
    # Print the data to the console (or log it)
    print("Received demo booking:")
    print(f"Email: {email}")
    print(f"Phone: {phone}")
    print(f"Date: {date}")
    print(f"Time Slot: {timeslot}")

    date_obj = datetime.strptime(date, "%Y-%m-%d")

    # Log the data
    logging.info(f"Received demo booking: name: {name}, Email: {email}, Phone: {phone}, Date: {date_obj}, Time Slot: {timeslot}, Message: {message}")

    existing_booking = BookingAppointment.query.filter_by(date=date_obj, timeslot=timeslot).first()
    print("existing_booking ===> ", existing_booking)
    if existing_booking:
        return jsonify({"error": "This time slot is already booked."}), 409

    # Create a new booking entry
    try:
        new_booking = BookingAppointment(email=email, phone=phone, date=date_obj, timeslot=timeslot, name=name, message=message)
        # Add the booking to the session and commit it to the database
        db.session.add(new_booking)
        db.session.commit()
    except Exception as e:
        print("error ===> ", e)

    # Prepare the email details for the user
    subject_user = "Demo Booking Confirmation"
    username = email.split('@')[0]  # Get username from email address
    booking_details = {
        'date': date,
        'timeslot': timeslot,
        'email': email,
        'phone': phone,
    }
    # # Send the demo booking confirmation email to the user
    try:
        send_demo_email([email], subject_user, username, booking_details)
    except Exception as e:
        print("error ===> ", e)
    # Prepare the email details for the site owner
    subject_owner = "New Demo Booking Received"
    owner_email = "vaibhavsharma3070@gmail.com"
    print("before sending email")

    # Respond with a success message
    return jsonify({"message": "Demo booking received successfully!"}), 200

@app.route('/available_time_slots', methods=['GET'])
def available_time_slots():
    # Get the date from the request arguments
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is required.'}), 400
    # Convert the date string to a datetime object=
    # Define all possible time slots
    all_time_slots = [
        "10:00 AM to 10:45 AM", 
        "11:00 AM to 11:45 AM", 
        "12:45 PM to 1:30 PM", 
        "1:45 PM to 2:30 PM", 
        "2:45 PM to 3:30 PM"
    ]
    # Fetch booked time slots from the database for the selected date

    try:
        date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
    except Exception as e:
        print("error ===> ", e)
    try:
        booked_slots = [booking.timeslot for booking in BookingAppointment.query.filter_by(date=date_obj).all()]
    except Exception as e:
        print("error ===> ", e)
    # Determine available time slots
    available_slots = [slot for slot in all_time_slots if slot not in booked_slots]
    return jsonify({'available_time_slots': available_slots}), 200


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=8501)
    socketio.run(app, host="0.0.0.0", port=8501)
