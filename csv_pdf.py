import pandas as pd
from fpdf import FPDF

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('SES_input_copy.csv')

# Create a PDF document
pdf = FPDF()
pdf.add_page()

# Set font for the table
pdf.set_font("Arial", size=12)

# Add a cell for each value in the DataFrame
for i in range(len(df)):
    for j in range(len(df.columns)):
        pdf.cell(40, 10, str(df.iloc[i, j]), border=1)
    pdf.ln()

# Save the PDF to a file
pdf.output("SES_CSV_output.pdf", 'F')