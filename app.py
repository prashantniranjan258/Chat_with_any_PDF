import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        else:
            print(f"Warning: No text found on page {i+1}")

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF. Ensure it has selectable text.")
    
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    return text_splitter.split_text(text)

def get_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks found! Ensure the PDF is correctly processed.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # ✅ Corrected
  
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # ✅ Corrected
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response.get("output_text", "No response generated."))



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say: "answer is not available in the context".
    Do not provide incorrect information.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-pro")  # Or gemini-1.5-flash
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply:", response.get("output_text", "No response generated."))

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with any PDF DOcument")
    
    user_question = st.text_input("Ask a question about the uploaded PDFs")
    if user_question:
        user_input(user_question)
    
    pdf_path = "espectra_2024_dec.pdf"  # Specify the path of the PDF file
    
    if os.path.exists(pdf_path):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Processing complete! You can now ask questions.")
    else:
        st.error("PDF file not found. Please check the file path.")

if __name__ == "__main__":
    main()
