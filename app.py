import os
import sys
import json 
import boto3
import streamlit as st
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    pdf_loader = PyPDFDirectoryLoader(path="data")
    pdf_documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents=pdf_documents)
    return docs 

def get_vector_store(docs):
    faiss_vector_store = FAISS.from_documents(documents=docs, embedding=bedrock_embeddings)
    faiss_vector_store.save_local("FAISS_Index")

def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={'max-tokens-to-sample':512})
    return llm

def get_jurassic_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end, 
but also include a summary of at least 250 words with detailed explanations. 
If you don’t know the answer, just say that you don’t know; don’t try to make up an answer.

<context>
{context}
</context

Question: {question}

Assistant:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response(llm, faiss_vector_store, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="📄",  
        layout="wide"
    )
    
    st.header("Chat with PDF using AWS Bedrock 💁")
    user_question = st.text_input("Ask a question from the PDF files!!")

    with st.sidebar:
        st.title("Create or Update Vectors: ")
        if st.button("Create/Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if user_question:
        try:
            faiss_index = FAISS.load_local("FAISS_Index", bedrock_embeddings)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return

        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                llm = get_claude_llm()
                st.write(get_response(llm, faiss_index, user_question))
                st.success("Done")

        if st.button("Llama2 Output"):
            with st.spinner("Processing..."):
                llm = get_llama2_llm()
                st.write(get_response(llm, faiss_index, user_question))
                st.success("Done")

        if st.button("Jurassic Output"):
            with st.spinner("Processing..."):
                llm = get_jurassic_llm()
                st.write(get_response(llm, faiss_index, user_question))
                st.success("Done")

if __name__ == "__main__":
    main()



