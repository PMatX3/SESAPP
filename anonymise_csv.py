import pandas as pd
import random
import string

# Load the CSV file
df = pd.read_csv('SES_input_copy.csv')

# Anonymize names with pseudonyms
df['First Name'] = ['Name' + str(i) for i in range(1, len(df)+1)]
df['Surname'] = df['First Name']

# Anonymize email addresses with masking
df['Email'] = ['u***r' + str(i) + '@example.com' for i in range(1, len(df)+1)]

# Anonymize Candidate Owner name with pseudonyms
df['Candidate Owner'] = ['Owner' + str(i) for i in range(1, len(df)+1)]

# Anonymize phone numbers with partial masking 
df['Home Phone'] = ['+1(***)***' + str(random.randint(1000,9999)) for _ in range(len(df))]
df['Mobile'] = df['Home Phone']

# Save the anonymized data to a new CSV file
df.to_csv('SES_output_anonymisr.csv', index=False)