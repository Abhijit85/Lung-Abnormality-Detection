

# load environment variable
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
import json  # Add this line to import the json module
import base64  # Add this line to import the base64 module
import boto3  # Add this line to import the boto3 module

load_dotenv()
mongo_uri = os.getenv('MONGO_URI')
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')

# Create a new client and connect to the server
client = MongoClient(mongo_uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"An error occurred while connecting to MongoDB: {e}")
    
    
# calls Amazon Bedrock to get a vector from either an image, text, or both
def get_multimodal_vector(input_image_base64=None, input_text=None):
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name="us-east-1",
            # Passing credentials during client creation
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token)

        request_body = {}
        if input_text:
            request_body["inputText"] = input_text
        if input_image_base64:
            request_body["inputImage"] = input_image_base64
        request_body["embeddingConfig"] = {"outputEmbeddingLength": 384}
        body = json.dumps(request_body)
        response = bedrock.invoke_model(
            body=body,
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get("embedding")
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# creates a vector from an image file path
def get_vector_from_file(file_path):
    with open(file_path, "rb") as image_file:
        input_image_base64 = base64.b64encode(image_file.read()).decode('utf8')

    vector = get_multimodal_vector(input_image_base64=input_image_base64)
    return vector


# Vectorize dataset and load it to MongoDB Atlas
db = client['AbnormalityDetectionDB']
coll = db['LungX-RayImage']

image_folder = 'LungX-RayImage'

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Function to split the name of the image file into conditions
def hyphen_split(a):
    if a.count("-") == 1:
        return a.split("-")[0]
    return "-".join(a.split("_", 2)[:2])

for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(image_folder, image_file)

    # Read the image from file as binary data
    with open(image_path, 'rb') as image_file_obj:
        image_data = image_file_obj.read()

    img_embedding = get_vector_from_file(image_path)
    condition = hyphen_split(image_file)
    image_document = {
        'filename': image_file,
        'condition': condition,
        'embedding': img_embedding,
        #'feedback': str(None)
    }

    coll.insert_one(image_document)
    print(f"Inserted: {image_file}")

print("All images inserted into MongoDB.")