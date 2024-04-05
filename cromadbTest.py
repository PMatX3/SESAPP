import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def cromadb_test(file_name,query):
    # df=pd.read_csv('the_oscar_award.csv')
    df=pd.read_csv(file_name)
    df.head()
    # print(df.head())
    # df['text'] = 'Candidate ID : ' + df['Candidate Id'] + 'Employer name is ' + df['Employer Name'] + 'Start data is ' + df['Start Date'] 
    # df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'
    # Create a new column 'text' concatenating values from each column

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

    # Print the first few rows to verify
    # print(df['text'].head())

    # print(df.head()['text'])
    # exit()
    import os
    import openai
    import chromadb
    from chromadb.utils import embedding_functions
    
    def text_embedding(text):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )


    client = chromadb.Client()
    collection = client.get_or_create_collection("oscars-2023",embedding_function=openai_ef)

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
    # print(res)
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
    # print("Last response : ",response_message)
    return response_message