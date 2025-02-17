from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import mimetypes
from pathlib import Path
from langchain.chains import RetrievalQA
import requests

import gradio as gr

def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn
warnings.filterwarnings('ignore')

def get_loader(filename):
    file_extension = Path(filename).suffix.lower()

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
        ".xml": UnstructuredMarkdownLoader
    }

    # return the loader class if found, else UnstructuredFileLoader
    loaderClass = loaders.get(file_extension, UnstructuredFileLoader)
    return loaderClass


def get_llm():
    model_id = "mistralai/mixtral-8x7b-instruct-v01"
    parameters = {
        GenParams.MAX_NEW_TOKENS : 256,
        GenParams.TEMPERATURE : 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id = model_id,
        params = parameters,
        url = "https://us-south.ml.cloud.ibm.com",
        project_id= project_id
    )
    return watsonx_llm

def document_loader(file):
    loaderClass = get_loader(file.name)
    loader = loaderClass(file.name)
    document = loader.load()
    return document


def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function = len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def watsonx_embedding():
    params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS:3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS:{"input_text":True}
    }

    watsonx_embedding = WatsonxEmbeddings(
        params = params,
        model_id = "ibm/slate-125m-english-rtrvr",
        project_id = "skills-network",
        url = "https://us-south.ml.cloud.ibm.com"
    )
    return watsonx_embedding

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    embeddings = vector_database(chunks)
    retriever = embeddings.as_retriever()

    return retriever

def retriever_qa(file, query):
    retriever_obj = retriever(file)
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(
        retriever = retriever_obj,
        llm = llm,
        chain_type = "stuff",
        return_source_documents=False
    )

    responce = qa.invoke(query)
    return responce['result']

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload Your File",file_count="single",file_types=['.pdf','.docx','.csv','.json'],type="filepath"),
        gr.Textbox(label="Input Query",lines=2,placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF file and ask any question from its content"
)

rag_application.launch(server_name="127.0.0.1",server_port=7860)


