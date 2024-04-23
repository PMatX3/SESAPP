from flask import Flask, request, render_template, flash, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os,json
from cromadbTest import load_data, execute_query, load_pdf_data, get_chat_history, load_json_data, get_chat_list, add_chat_message
from utils import get_pdf_text, get_text_chunks
import pandas as pd
import uuid
import threading
import requests
from config import BASE_URL
from htmlTemplates import css, bot_template, user_template

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///passwords.db'
db = SQLAlchemy(app)

app.secret_key = '31dee9b85d3be7513eda6e3bb1b2e22edd923194d18b3cf8'


class Password(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    login_attempts = db.Column(db.Integer, default=0)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

def load_and_cache_json_data():
    def load_json():
        with open('candidates_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        load_json_data(data)

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_json)
    thread.start()

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
            else:
                password_from_api = None
                login_attempt = None
                
            if login_attempt == 0:
                return redirect('change_password')
            if password_from_api:
                if password_from_api == password:
                    # Generate a new chat_id using uuid
                    chat_id = str(uuid.uuid4())
                    session['user_id'] = app_name
                    session['chat_id'] = chat_id
                    return redirect('/')
                else:
                    return jsonify({'message': 'Invalid credentials'}), 401
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
    elif request.method == 'GET':
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
    load_and_cache_json_data()
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
            return jsonify({'message': f'PDF {filename} processed successfully'}), 200
        elif data_type == 'csv':
            # CSV processing logic
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
            # Assuming load_data function exists and is intended to process the DataFrame
            print(file_path)
            load_data(file_path)  # Process the DataFrame as needed
            # Further processing can be done here, similar to app.py, if necessary
            return jsonify({'message': f'CSV {filename} processed successfully'}), 200
        session['recrutly_id'] = False
        return jsonify({'message': f'File {filename} uploaded successfully', 'type': data_type}), 200
    else:
        session['recrutly_id'] = True
        return jsonify({'message': 'SES data processed.', 'type': data_type}), 200


@app.route('/ask', methods=['POST'])
def ask():
    with app.app_context():
        user_question = request.form.get('question')
        # Retrieve user_id and chat_id from the session
        user_id = session.get('user_id')
        chat_id = session.get('chat_id')
        recrutly_id = session.get('recrutly_id')
        
        # Ensure user_id and chat_id are available
        if not user_id or not chat_id:
            return jsonify({'error': 'User ID or Chat ID missing from session'}), 400
        if recrutly_id:
            response = execute_query(user_question, 'test1', True)
        else:
            response = execute_query(user_question, 'test1')

        add_chat_message(user_id, user_question, response, chat_id)
        data = {
            "user": user_question,
            "ai": response
        }
        return jsonify(data)

@app.route('/get_chat_history', methods=['GET'])
def api_get_chat_history():
    with app.app_context():
        print('Session Data:', dict(session))
        user_id = session.get('user_id')
        chat_id = session.get('chat_id')
        
        print('user id---->>>',user_id)
        print('chat id---->>>',chat_id)
        # Validate input
        if not user_id or not chat_id:
            return jsonify({'error': 'Missing user_id or chat_id'}), 400
        
        # Fetch chat history
        chat_history = get_chat_history(user_id, chat_id)
        
        # Check if chat history is found
        if chat_history:
            # Assuming chat_history is a list of dicts with 'message' and 'response' keys
            formatted_history = [{'user': user_template.replace('{{MSG}}',message['message']), 'ai': bot_template.replace('{{MSG}}',message['response'])} for message in chat_history]
            return jsonify(formatted_history), 200
        else:
            return jsonify({'error': 'Chat history not found'}), 404

@app.route('/store_password', methods=['POST'])
def store_password():
    with app.app_context():
        data = request.form
        app_name = data.get('app_name')
        password = data.get('password')
        new_password = Password(app_name=app_name, password=password)
        db.session.add(new_password)
        db.session.commit()
        return 'Password stored successfully'

@app.route('/get_password', methods=['GET'])
def get_password():
    with app.app_context():
        app_name = request.args.get('app_name')
        
        stored_password = Password.query.filter_by(app_name=app_name).first()
        
        if stored_password:
            # Return password and login attempts as a JSON response
            return jsonify({
                'password': stored_password.password,
                'login_attempts': stored_password.login_attempts
            }), 200
        else:
            # Return a JSON response indicating that no password was found
            return jsonify({'error': 'Password not found'}), 404

@app.route('/update_password', methods=['POST'])
def update_password():
    with app.app_context():
        data = request.form
        app_name = data.get('app_name')
        new_password = data.get('new_password')
        password_record = Password.query.filter_by(app_name=app_name).first()
        if password_record:
            password_record.password = new_password
            db.session.commit()
            requests.post(url_for('update_login_attempts', _external=True), data={'app_name': app_name})
            return 'Password updated successfully'
        else:
            return 'Password record not found'

@app.route('/update_login_attempts', methods=['POST'])
def update_login_attempts():
    with app.app_context():
        data = request.form
        app_name = data.get('app_name')
        increment = data.get('increment', type=int, default=1)  # Optional: allows specifying increment amount, defaults to 1
        password_record = Password.query.filter_by(app_name=app_name).first()
        if password_record:
            if increment != 0:
                password_record.login_attempts += increment
            else:
                password_record.login_attempts = 0

            db.session.commit()
            return 'Login attempts updated successfully'
        else:
            return 'Password record not found'

@app.route('/get_all_credentials', methods=['GET'])
def get_all_credentials():
    with app.app_context():
        passwords = Password.query.all()
        app_names = [password.app_name for password in passwords]
        passwords_list = [password.password for password in passwords]
        return jsonify({'app_names': app_names, 'passwords': passwords_list})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5050)