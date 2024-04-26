from flask import Flask, request, render_template, flash, jsonify, session, redirect, url_for
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os,json
from cromadbTest import load_data, execute_query, load_pdf_data, get_chat_history, load_json_data, get_chat_list, add_chat_message
from utils import get_pdf_text, get_text_chunks
import pandas as pd
import uuid
import threading
import requests
from config import BASE_URL
from datetime import datetime
import markdown
from htmlTemplates import css, bot_template, user_template

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/user_db"

client = MongoClient(app.config["MONGO_URI"])
mongo = client['user_db']

app.secret_key = '31dee9b85d3be7513eda6e3bb1b2e22edd923194d18b3cf8'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_cache_json_data(filename=None):
    def load_json():
        if filename:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                load_json_data(data, True)
        else:
            with open('candidates_data.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
                load_json_data(data)
        

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_json)
    thread.start()

def calculate_days_since_join(join_date_str):
    join_date = datetime.strptime(join_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    difference = current_date - join_date
    return difference.days

@app.route('/')
def index():
    if 'user_id' in session:
        days_since_join = session['days_since_join'] 
        days = session['days']
        if days_since_join >= days:
            return redirect(url_for('pricing'))
        else:
            return render_template('index.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

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
        print("Rendering admin with users:", session['user_type'])
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

            if not app_name or not password:
                flash('Missing required user data', 'error')
                return render_template('insert.html'), 400

            # Get the current date and convert to string
            join_date = datetime.now().date().isoformat()

            # Insert the new user into the MongoDB
            mongo.users.insert_one({
                'app_name': app_name,
                'password': password,
                'login_attempts': int(login_attempts),
                'days': int(days),
                'ses': True,
                'type': user_type,  # Store the user type
                'join_date': join_date  # Store the join date as string
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
        new_ses = request.form.get('new_ses', type=bool)
        new_usertype = request.form.get('user_type')

        update_fields = {
            'app_name': new_data,
            'password': new_password,
            'days': new_days,
            'login_attempts': new_login_attempts,
            'ses': new_ses,
            'type': new_usertype
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
    if delete_result.deleted_count > 0:
        flash('Record deleted successfully!')
    else:
        flash('No record found with the specified app name.')
    return redirect(url_for('admin'))

@app.route('/updated_cmp', methods=['POST'])
def updated_cmp():
    data = request.json
    app_name = data.get('app_name')
    ses = data.get('ses')
    
    if not app_name or ses is None:
        return jsonify({'error': 'Missing app_name or ses'}), 400

    update_result = mongo.users.update_one(
        {'app_name': app_name},
        {'$set': {'ses': ses}}
    )

    if update_result.modified_count > 0:
        return jsonify({'message': 'SES updated successfully'}), 200
    else:
        return jsonify({'error': 'No record found with the specified app_name or update failed'}), 404

# user part
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        with app.app_context():
            app_name = request.form.get('app_name')
            password = request.form.get('password')
            response = requests.get(f'{BASE_URL}/get_password?app_name={app_name}')
            if response.status_code == 200:
                data = response.json()
                password_from_api = data['password']
                login_attempt = data['login_attempts']
                days = data['days']
                join_date = data['join_date']
                user_type = data['type']
                ses = data['ses']
                days_since_join = calculate_days_since_join(join_date)
                print('days since ---- >>>> ',days_since_join)
            else:
                password_from_api = None
                login_attempt = None
                days = None
                user_type = None
                days_since_join = 0
                ses = None
            session['days_since_join'] = days_since_join
            session['days'] = days
            session['ses'] = ses
            session['user_type'] = user_type
            if login_attempt == 0:
                print('herer')
                return jsonify({'login_attempt':login_attempt,'days_since_join':days_since_join})
            
            if password_from_api:
                if password_from_api == password:
                    # Generate a new chat_id using uuid
                    chat_id = str(uuid.uuid4())
                    session['user_id'] = app_name
                    session['chat_id'] = chat_id  
                    if user_type == 'admin':
                        return jsonify({'user':'admin'})
                    else:
                        return jsonify({'user':'user'})
                else:
                    return jsonify({'message': 'Invalid credentials'}), 401
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
    elif request.method == 'GET':
        session.clear()
        return render_template('login.html')

@app.route('/logout')
def logout():
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
                print(response.json)
                if response.status_code == 200:
                    chat_history = response.json
                else:
                    chat_history = {'error': 'Failed to retrieve chat history'}
                return jsonify({'chat_list': chat_list, 'chat_history': chat_history})
        
        return jsonify({'chat_list': chat_list, 'chat_history': chat_history})

@app.route('/new_chat')
def new_chat():
    # Generate a new chat_id using uuid
    chat_id = str(uuid.uuid4())
    # Set the new chat_id in the session
    session['chat_id'] = chat_id
    # Render the index.html page
    return render_template('index.html')

@app.route('/change_password')
def change_password():
    return render_template('change_password.html')

@app.route('/load_data', methods=['POST'])
def load_new_data():
    try:
        file = request.files['file']
    except:
        file = None
    data_type = request.form.get('type')
    if data_type != 'json':
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if data_type == 'pdf':
            # Assuming get_pdf_text and get_text_chunks functions are defined similarly to those in app.py
            pdf_docs = [file_path]  # Adjusted to work with a single file
            raw_text = get_pdf_text(pdf_docs)  # Extract text from the PDF
            text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
            # Assuming load_pdf_data function exists and is intended to process the raw text or chunks
            load_pdf_data(raw_text)  # Process the extracted text as needed
            # Further processing can be done here, similar to app.py, if necessary
            return jsonify({'message': 'File Proceed', 'type': data_type}), 200
        elif data_type == 'csv':
            # CSV processing logic
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
            # Assuming load_data function exists and is intended to process the DataFrame
            print(file_path)
            load_data(file_path)  # Process the DataFrame as needed
            # Further processing can be done here, similar to app.py, if necessary
            return jsonify({'message': 'File Proceed', 'type': data_type}), 200
        elif data_type == 'jsonfile':
            #firstly save file in uploads 
            # with open(file_path, 'r') as file:
            #     data = file.read()
            
            # json_data = json.loads(data)
            starttime = datetime.now()
            load_and_cache_json_data(file_path)
            endtime = datetime.now()
            print(f"Time taken: {endtime - starttime}")
            # load_json_data(json_data, True)
            return jsonify({'message': 'File Proceed', 'type': data_type}), 200
        session['recrutly_id'] = False
        return jsonify({'message': 'File Proceed', 'type': data_type}), 200
    else:
        load_and_cache_json_data()
        session['recrutly_id'] = True
        return jsonify({'message': 'SES data processed.', 'type': data_type}), 200


@app.route('/ask', methods=['POST'])
def ask():
    with app.app_context():
        user_question = request.form.get('question')
        # Retrieve user_id and chat_id from the session
        user_id = session.get('user_id')
        chat_id = session.get('chat_id')
        recruitly_data = session.get('recruitly-data')
        # Ensure user_id and chat_id are available
        if not user_id or not chat_id:
            return jsonify({'error': 'User ID or Chat ID missing from session'}), 400
        if recruitly_data:
            response = execute_query(user_question, 'test1', True)
        else:
            response = execute_query(user_question, 'test1')

        add_chat_message(user_id, user_question, markdown.markdown(response), chat_id)
        data = {
            "user": user_question,
            "ai": response
        }
        return jsonify(data)

@app.route('/get_chat_history', methods=['GET'])
def api_get_chat_history():
    with app.app_context():
        user_id = session.get('user_id')
        chat_id = session.get('chat_id')
    
        if not user_id or not chat_id:
            return jsonify({'error': 'Missing user_id or chat_id'}), 400
        
        chat_history = get_chat_history(user_id, chat_id)
        
        if chat_history:
            for message in chat_history:
                message['_id'] = str(message['_id'])  # Convert ObjectId to string
            formatted_history = [{'user': user_template.replace('{{MSG}}', message['message']), 'ai': bot_template.replace('{{MSG}}', message['response'])} for message in chat_history]
            return jsonify(formatted_history), 200
        else:
            return jsonify({'error': 'Chat history not found'}), 404

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

if __name__ == '__main__':
    app.run(debug=True, port=5050)