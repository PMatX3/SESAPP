from flask import Flask, request, render_template, flash, jsonify, session, redirect, url_for, Response
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os,json
from cromadbTest import load_data, execute_query, load_pdf_data, get_chat_history, load_json_data, get_chat_list, add_chat_message
from utils import get_pdf_text, get_text_chunks, send_reset_password_mail
import pandas as pd
import uuid
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
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/user_db"

stripe_keys = {
    "secret_key": os.environ["STRIPE_SECRET_KEY"],
    "publishable_key": os.environ["STRIPE_PUBLISHABLE_KEY"],
}

stripe.api_key = stripe_keys["secret_key"]

client = MongoClient(app.config["MONGO_URI"])
mongo = client['user_db']

app.secret_key = '31dee9b85d3be7513eda6e3bb1b2e22edd923194d18b3cf8'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DOMAIN = 'https://www.yourbestcandidate.ai'

md = (
    MarkdownIt('commonmark' ,{'breaks':True,'html':True})
    .use(front_matter_plugin)
    .use(footnote_plugin)
    .enable('table')
)

def load_and_cache_json_data(filename=None, temp=False):
    def load_json():
        if filename and temp:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                load_json_data(data, True)
        else:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                load_json_data(data)
        

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_json)
    thread.start()

def load_and_cache_file_data(filename=None):
    def load_file():
        load_data(filename)
        

    # Start a new thread for loading JSON data
    thread = threading.Thread(target=load_file)
    thread.start()

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
            break

    if file_found:
        load_and_cache_json_data(file_found)
        return 200
    else:
        print("No file found containing the company name in the filename.")
        return 'You don\'t have any data loaded into the database'

def calculate_days_since_join(join_date_str):
    join_date = datetime.strptime(join_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    difference = current_date - join_date
    return difference.days

@app.route('/')
def index():
    if 'user_id' in session:
        user_id = session['user_id']
        user_data = mongo.users.find_one({'app_name': user_id})
        if user_data:
            days_since_join = calculate_days_since_join(user_data['join_date'])
            days = user_data.get('days', 0)
            login_attempts = user_data.get('login_attempts')
            session['days_since_join'] = days_since_join
            session['days'] = days
            session['login_attempt'] = login_attempts

            if days_since_join >= days:
                return redirect(url_for('pricing'))
            else:
                return render_template('index.html')
        else:
            return redirect(url_for('login'))
    else:
        return redirect(url_for('login'))

#PAYMENT PART    
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route("/charge", methods=["POST"])
def charge():
    amount = 1000  # amount in cents
    currency = "usd"
    description = "A Flask Charge"

    stripe.api_key = stripe_keys["secret_key"]

    create_checkout_session = stripe.checkout.Session.create(
        line_items=[
            {
                'price':'price_1PAm40DJlqPYNFXOr648BpiK',
                'quantity':1
            }
        ],
        mode='subscription',
        success_url=f'{DOMAIN}/',
        cancel_url=f'{DOMAIN}/pricing'
    )

    if create_checkout_session['payment_status'] == 'unpaid':
        with app.test_client() as client:
            response = client.post('/edit/' + session['user_id'], data={'new_days': 100, 'new_join_date': datetime.now().date().isoformat()})
            if response.status_code == 200:
                print('Days updated to 100 for unpaid session.')
            else:
                print('Failed to update days for unpaid session.')

    return redirect(create_checkout_session.url,303)

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
        send_reset_password_mail(user_id,'Reset Password link', firstname, reset_link)
        return jsonify({'message':'email sent!'})
    else:
        return render_template('reset_password.html')

#AI Part
@app.route('/load_data', methods=['POST'])
def load_new_data():
    try:
        files = request.files
    except:
        files = None
    data_types = request.form
    data_types = data_types.getlist('type')
    print(files)
    message = None
    if files:
        files_list = files.getlist('file')
        for i,file in enumerate(files_list):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            _, file_ext = os.path.splitext(filename)  # Extract the file extension
            if file_ext == '.pdf':
                pdf_docs = [file_path]  # Adjusted to work with a single file
                raw_text = get_pdf_text(pdf_docs)  # Extract text from the PDF
                text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
                load_pdf_data(raw_text)  # Process the extracted text as needed
            elif file_ext == '.csv':
                # CSV processing logic
                df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
                load_and_cache_file_data(file_path)
                time.sleep(20)
            elif file_ext == '.json':
                starttime = datetime.now()
                load_and_cache_json_data(file_path, True)
                endtime = datetime.now()
                print(f"Time taken: {endtime - starttime}")
        session['recrutly_id'] = False
        message = "File Successfully Processed"

    if 'json' in data_types:
        user_id = session.get('user_id')
        session['recrutly_id'] = True
        user = mongo.users.find_one({'app_name': user_id})
        if user:
            company_name = user.get('company', 'No company associated')
        
        res = check_for_file(company_name)
        
        time.sleep(20)
        if message is not None:
            message += ' and '+f'{company_name} data processed.'
        else:
            message = f'{company_name} data processed.'
        if res != 200:
           message = res 
           return message, 404
    return jsonify({'message':message}), 200


# @app.route('/ask', methods=['POST'])
# def ask():
#     user_question = request.form.get('question')
#     user_id = session.get('user_id')
#     chat_id = session.get('chat_id')
#     recruitly_data = request.form.get('recruitly_data', False)

#     if not user_id or not chat_id:
#         return jsonify({'error': 'User ID or Chat ID missing from session'}), 400
#     print(recruitly_data)
#     # Call execute_query and stream the results, rendering them as HTML
#     def generate():
#         entire_response = ''
#         for chunk in execute_query(user_question, user_id, recruitly_data == 'true'):
#             entire_response += chunk
#             yield md.render(entire_response)
#         add_chat_message(user_id,user_question,md.render(entire_response),chat_id)

#     return Response(generate(), mimetype='text/html')
@socketio.on('ask')
def handle_ask(json):
    user_question = json['question']
    user_id = session.get('user_id')
    chat_id = session.get('chat_id')
    recruitly_data = json.get('recruitly_data', False)

    if not user_id or not chat_id:
        emit('error', {'error': 'User ID or Chat ID missing from session'})
        return

    entire_response = ''
    for chunk in execute_query(user_question, user_id, recruitly_data):
        entire_response += chunk
        emit('message', {'data': md.render(entire_response)})

    add_chat_message(user_id, user_question, md.render(entire_response), chat_id)

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

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=8501)
    socketio.run(app, host="0.0.0.0", port=8501)