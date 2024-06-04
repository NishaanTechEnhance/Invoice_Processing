import os
from flask import Flask, render_template, request, jsonify, redirect
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Initialize Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=os.getenv('GOOGLE_API_KEY'))

# Configure Azure Cosmos DB
COSMOS_DB_URI = os.getenv('COSMOS_DB_URI')
COSMOS_DB_KEY = os.getenv('COSMOS_DB_KEY')
COSMOS_DB_DATABASE = os.getenv('COSMOS_DB_DATABASE')
COSMOS_DB_CONTAINER = os.getenv('COSMOS_DB_CONTAINER')

client = CosmosClient(COSMOS_DB_URI, COSMOS_DB_KEY)
database = client.get_database_client(COSMOS_DB_DATABASE)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Configure Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def upload_to_blob_storage(file, filename):
    blob_client = blob_container_client.get_blob_client(filename)
    # Set overwrite=True to overwrite existing blobs
    blob_client.upload_blob(file, overwrite=True)
    blob_url = blob_client.url
    return blob_url

# Function to process image using the model
def process_image_with_model(blob_url):
    hmessage1 = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract the following data: invoice number, invoice date, due date, customer name, customer address, total amount.",
            },
            {"type": "image_url", "image_url": blob_url},
        ]
    )

    message1 = llm.invoke([hmessage1])
    extracted_data = {}
    content_string = message1.content
    lines = content_string.split("\n")
    for line in lines:
        if line.strip():
            key, value = line.split(":", 1)
            extracted_data[key.strip()] = value.strip()

    # Store the extracted data in Azure Cosmos DB
    container.upsert_item({
        "id": str(extracted_data.get("Invoice Number", "")),  # Use a unique identifier for the document
        "invoice_number": extracted_data.get("Invoice Number", ""),
        "invoice_date": extracted_data.get("Invoice Date", ""),
        "due_date": extracted_data.get("Due Date", ""),
        "customer_name": extracted_data.get("Customer Name", ""),
        "customer_address": extracted_data.get("Customer Address", ""),
        "total_amount": extracted_data.get("Total Amount", ""),
        "blob_url": blob_url  # Store the URL of the uploaded image
    })

    return extracted_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        blob_url = upload_to_blob_storage(file, file.filename)
        processed_data = process_image_with_model(blob_url)
        return jsonify(processed_data)
    return redirect(request.url)

if __name__ == "__main__":
    
    app.run(debug=True)
