import os
import json
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
s3_client = boto3.client('s3')

BUCKET_NAME = "customer-support-faq"  
PDF_S3_PREFIX = "data/"
FAISS_INDEX_S3_PATH = "faiss_index/index.faiss"
FAISS_INDEX_LOCAL_PATH = "/tmp/faiss_index"

prompt_template = """
Human: Answer the question in one paragraph using natural, conversational language.
Write as if explaining to a colleague - avoid formal phrases like "the policy states".
Just provide the direct answer.<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def data_ingestion():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PDF_S3_PREFIX)
    docs = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('.pdf'):
            local_path = f"/tmp/{os.path.basename(key)}"
            s3_client.download_file(BUCKET_NAME, key, local_path)
            if not os.path.exists(local_path):
                print(f"Failed to download {key} to {local_path}")
                raise FileNotFoundError(f"Could not download {key} to {local_path}")
            else:
                print(f"Successfully downloaded {key} to {local_path}")
            loader = PyPDFLoader(local_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            docs.extend(text_splitter.split_documents(documents))
            os.remove(local_path)  
    if not docs:
        raise ValueError("No PDFs found in s3://{}/{}".format(BUCKET_NAME, PDF_S3_PREFIX))
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    if not os.path.exists(FAISS_INDEX_LOCAL_PATH):
        os.makedirs(FAISS_INDEX_LOCAL_PATH)
    vectorstore_faiss.save_local(FAISS_INDEX_LOCAL_PATH)
    s3_client.upload_file(
        os.path.join(FAISS_INDEX_LOCAL_PATH, "index.faiss"),
        BUCKET_NAME,
        FAISS_INDEX_S3_PATH
    )
    s3_client.upload_file(
        os.path.join(FAISS_INDEX_LOCAL_PATH, "index.pkl"),
        BUCKET_NAME,
        "faiss_index/index.pkl"  # Upload index.pkl to S3
    )

def download_vector_store_from_s3():
    if not os.path.exists(FAISS_INDEX_LOCAL_PATH):
        os.makedirs(FAISS_INDEX_LOCAL_PATH)
    local_file = os.path.join(FAISS_INDEX_LOCAL_PATH, "index.faiss")
    s3_client.download_file(BUCKET_NAME, FAISS_INDEX_S3_PATH, local_file)
    s3_client.download_file(
        BUCKET_NAME,
        "faiss_index/index.pkl",
        os.path.join(FAISS_INDEX_LOCAL_PATH, "index.pkl")
    )
def get_claude_llm():
    llm = BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={'max_tokens': 512}  # Note: `maxTokens` becomes `max_tokens` for BedrockChat
    )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})['result']

def lambda_handler(event, context):
    print("RAW_EVENT:", event)  # Add this
    print("EVENT:", json.dumps(event, indent=2))
    with open("/tmp/event_debug.json", "w") as f:
            json.dump(event, f)
        # Handle ingestion
    if event.get("operation") == "ingest":
        try:
            docs = data_ingestion()
            get_vector_store(docs)
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "FAISS index created and uploaded to S3"})
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Ingestion failed: {str(e)}"})
            }

    # Handle query
    try:
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=FAISS_INDEX_S3_PATH)
        except Exception:
            # Si l'index n'existe pas, lancer l'ingestion
            docs = data_ingestion()
            get_vector_store(docs)
        query = event['sessionState']['intent']['slots']['Query']['value']['interpretedValue']
       
    except (KeyError, TypeError):
        return {
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": "Failed",
                "message": {
                    "contentType": "PlainText",
                    "content": "I'm sorry, I couldn't understand your query. Please try again."
                }
            }
        }

    try:
        download_vector_store_from_s3()
        faiss_index = FAISS.load_local(FAISS_INDEX_LOCAL_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return {
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": "Failed",
                "message": {
                    "contentType": "PlainText",
                    "content": f"Error loading FAISS index: {str(e)}"
                }
            }
        }

    try:
        llm = get_claude_llm()
        answer = get_response_llm(llm, faiss_index, query)
    except Exception as e:
        return {
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": "Failed",
                "message": {
                    "contentType": "PlainText",
                    "content": f"Error processing your query: {str(e)}"
                }
            }
        }

    response = {
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": "Fulfilled",
            "message": {
                "contentType": "PlainText",
                "content": answer
            }
        }
    }

    if 'sessionAttributes' in event.get('sessionState', {}):
        response["sessionAttributes"] = event['sessionState']['sessionAttributes']

    return response