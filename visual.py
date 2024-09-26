import streamlit as st
import pandas as pd
import openai
import plotly.graph_objects as go
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re

# Initialize LLM
def initialize_llm(api_key):
    openai.api_key = api_key
    return OpenAI(openai_api_key=api_key, temperature=0.9, max_tokens=500)

# Load Excel files
def load_excel_file(file):
    try:
        return pd.read_excel(file, sheet_name=None)
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Process sheets into documents
def process_sheets_to_documents(sheet_dict):
    documents = []
    for sheet_name, df in sheet_dict.items():
        for index, row in df.iterrows():
            content = f"{sheet_name}: " + ", ".join(str(value) for value in row.values)
            documents.append(Document(page_content=content, metadata={"sheet": sheet_name}))
    return documents

# Split documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Create vector index
def create_vector_index(docs, api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.from_documents(docs, embeddings)

# Custom prompt template for better control
prompt_template_text = """
Provide precise and accurate answers based solely on the information explicitly stated in the financial documents.
Ensure 100% accuracy, with no assumptions or inferences.

Question: {query}

Answer the question directly based on the provided documents. If the answer is not available, respond with "I don't know".
"""
custom_prompt_template = PromptTemplate(input_variables=["query"], template=prompt_template_text)

# Run QA chain with the custom prompt
def run_qa_chain(query, vectorindex, llm):
    retriever = vectorindex.as_retriever()
    chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, prompt_template=custom_prompt_template)
    qa_results = chain({"query": query})
    return qa_results

def extract_chart_title(query):
    entity_pattern = r"(Company|Organization|Firm|Business) ([\w\s]+)"
    metric_pattern = r"(revenue|sales|profit|earnings|income)"
    time_pattern = r"last (\d+) years?"

    try:
        entity = re.search(entity_pattern, query).group(2)
        metric = re.search(metric_pattern, query).group(0)
        time = re.search(time_pattern, query).group(1)
        return f"{metric.capitalize()} of {entity} Over Last {time} Years"
    except AttributeError:
        return "Unable to extract chart title"

# Streamlit UI
st.title("Business Data Visualization Tool")
st.write("Upload multiple Excel files to query data and visualize it.")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# Initialize LLM
if api_key:
    llm = initialize_llm(api_key)

# File uploader for Excel files
uploaded_files = st.file_uploader("Choose Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files:
    all_dataframes = {}
    for uploaded_file in uploaded_files:
        sheets_dict = load_excel_file(uploaded_file)
        documents = process_sheets_to_documents(sheets_dict)
        for sheet_name, df in sheets_dict.items():
            all_dataframes[sheet_name] = df

    docs = split_documents(documents)
    vectorindex_openai = create_vector_index(docs, api_key)

    query = st.text_input("Enter your query (e.g., 'Show me revenue for Company XYZ'): ")

    if query:
        try:
            qa_results = run_qa_chain(query, vectorindex_openai, llm)
            result = qa_results['result']
            metric = extract_chart_title(query)

            data_matches = [int(re.sub(r'[\$,\.:]', '', x)) for x in result.split('was')[1].replace('and', '').split()]

            years = []
            for doc in documents:
                if 'Date' in doc.page_content:
                    dates = doc.page_content.split(', ')[1:]
                    years = [date.split('-')[0] for date in dates]
            years = years[-len(data_matches):]

            sorted_data = sorted(zip(data_matches, years), reverse=True)
            data, years = zip(*sorted_data)
            fig = go.Figure(data=[go.Bar(x=years, y=data, hovertext=[f"{year}: {format(x, ',d')}" for x, year in zip(data, years)], hoverinfo='text')])
            fig.update_layout(title=f'{metric.capitalize()} Over Last {len(data)} Years', xaxis_title='Year', yaxis_title=f'{metric.capitalize()} (in billions)')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing query: {e}")
