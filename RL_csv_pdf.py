from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('SES_input_copy.csv')

# Create a PDF document
pdf_filename = "RL_SES_output.pdf"
pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
elements = []

# Convert the DataFrame to a list of lists for the table
data = [df.columns.tolist()] + df.values.tolist()

# Create a table from the data
table = Table(data)

# Add style to the table
'''
style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)])
table.setStyle(style)
'''
# Add the table to the elements list
elements.append(table)

# Build the PDF
pdf.build(elements)

print(f"PDF created: {pdf_filename}")