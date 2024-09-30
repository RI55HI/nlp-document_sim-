import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import pandas as pd

# Function to load pre-trained BERT model and tokenizer
@st.cache_resource
def load_model():
    st.write("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    st.write("Model and tokenizer loaded successfully!")
    return tokenizer, model

# Function to read text from docx file
def read_docx(file):
    return docx2txt.process(file)

# Function to read text from PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to read text from Excel file
def read_excel(file):
    df = pd.read_excel(file)
    text = df.to_string()  # Convert the Excel data to a string
    return text

# Function to handle different file types and extract text
def extract_text(file):
    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    elif file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return read_excel(file)
    else:
        return file.getvalue().decode("utf-8")  # For text files

# Function to calculate embeddings using BERT
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()

# Function to calculate similarity between two documents
def calculate_similarity(doc1, doc2, tokenizer, model):
    embedding1 = get_embeddings(doc1, tokenizer, model)
    embedding2 = get_embeddings(doc2, tokenizer, model)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0] * 100  # Return similarity as percentage

# Streamlit UI
def main():
    st.title("Document Similarity Detection with BERT")
    
    # Display initial instructions
    st.write("This app calculates the similarity between two documents (txt, docx, pdf, xlsx) using BERT embeddings. Please upload two documents for comparison.")
    
    # User input: upload two files
    file1 = st.file_uploader("Upload the first document", type=["txt", "docx", "pdf", "xlsx"])
    file2 = st.file_uploader("Upload the second document", type=["txt", "docx", "pdf", "xlsx"])

    # Ensure both files are uploaded before proceeding
    if file1 and file2:
        st.write(f"First file uploaded: {file1.name}")
        st.write(f"Second file uploaded: {file2.name}")

        # Load the BERT model and tokenizer
        tokenizer, model = load_model()

        if st.button("Calculate Similarity"):
            with st.spinner('Extracting text and calculating similarity...'):
                try:
                    # Extract text from both files
                    doc1 = extract_text(file1)
                    doc2 = extract_text(file2)

                    # Calculate similarity
                    similarity_percentage = calculate_similarity(doc1, doc2, tokenizer, model)
                    st.success(f"The similarity between the documents is: {similarity_percentage:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload both documents to proceed.")

if __name__ == "__main__":
    main()
