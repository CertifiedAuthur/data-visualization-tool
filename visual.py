import streamlit as st
import pandas as pd
import openai
import plotly.graph_objects as go
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re

# Function to initialize LLM with the provided OpenAI API key
def initialize_llm(api_key):
    openai.api_key = api_key  # Set the OpenAI API key globally
    return OpenAI(openai_api_key=api_key, temperature=0.5, max_tokens=1000)

# Function to load Excel files
def load_excel_file(file):
    """Load Excel file into a dictionary of DataFrames"""
    try:
        return pd.read_excel(file, sheet_name=None)
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Function to process sheets into documents
def process_sheets_to_documents(sheet_dict):
    """Process Excel sheets into LangChain documents"""
    documents = []
    for sheet_name, df in sheet_dict.items():
        for index, row in df.iterrows():
            content = f"{sheet_name}: " + ", ".join(str(value) for value in row.values)
            documents.append(Document(page_content=content, metadata={"sheet": sheet_name}))
    return documents

# Function to split documents for better processing
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Function to create vector index from documents
def create_vector_index(docs, api_key):
    """Create FAISS vector index from documents"""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.from_documents(docs, embeddings)

# Function to run query on the documents
def run_qa_chain(query, vectorindex, llm):
    """Run Retrieval QA chain on the documents"""
    retriever = vectorindex.as_retriever()
    chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
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

        chart_title = f"{metric.capitalize()} of {entity} Over Last {time} Years"
        return chart_title
    except AttributeError:
        return "Unable to extract chart title"
    
prompt ="""
You are AnalytIQ, a conversational data visualization assistant for financial and business analysts. Excel files will be provided to you, some will be in a format that is complex, but you must thoroughly search through it and provide answer to every query asked. 
Kindly make sure that your answer is 100% correct to what you have on the excel file especially when it comes to date and figures, do not misplace either of them, this is really important. please skip missing values, make sure to thoroughly go through the document meticulously and return the answer.
Your job is to assist financial and business analysts by providing insights, generating visual representations, and answering queries using data from multiple projects. You help users extract information, create visualizations, and interact with complex business data using natural language queries in a seamless and intuitive manner.
Return the results in a Sequential Numeric List Format. List all numeric values sequentially, separated by commas, and use 'and' before the last value.
Remember you are AnalytIQ, a conversational data visualization assistant for financial and business analysts.
"""

# Streamlit UI
st.title("Business Data Visualization Tool")
st.write("Upload multiple Excel files to query data and visualize it.")

# Load OpenAI API key
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# Initialize LLM only if API key is provided
if api_key:
    llm = initialize_llm(api_key)

# File uploader for multiple Excel files
uploaded_files = st.file_uploader("Choose Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files:
    all_dataframes = {}
    sheet_names = []

    # Process each uploaded Excel file
    for uploaded_file in uploaded_files:
        sheets_dict = load_excel_file(uploaded_file)
        documents = process_sheets_to_documents(sheets_dict)
        for sheet_name, df in sheets_dict.items():
            all_dataframes[sheet_name] = df
            sheet_names.append(sheet_name)

    docs = split_documents(documents)
    vectorindex_openai = create_vector_index(docs, api_key)

    # User query input
    query = st.text_input("Enter your query (e.g., 'Show me revenue for Company XYZ'): ")

    if query:
        try:
            # Run the query and get the response
            qa_results = run_qa_chain(prompt + query, vectorindex_openai, llm)
            result = qa_results['result']
            # print (result)
            # st.write(f"Query Result: {result}")  # Display the raw result

            # Check if the result is visualizable by searching for multiple numeric values
            # Extract numeric values from the result
            data_matches = []

            # Use regex to find all numeric values
            found_numbers = re.findall(r'(\d{9,})', result)

            # Convert found numbers to integers and add to data_matches
            data_matches = [int(x) for x in found_numbers]

            # Print the extracted data
            if data_matches:
                data_list = ', '.join(map(str, data_matches[:-1])) + ', and ' + str(data_matches[-1])
                print(f"Extracted Revenue Data: {data_list}")  # Display extracted data
            else:
                print("No valid numeric data found.")


            # If we successfully extract multiple data points, visualize
            if len(data_matches) > 1:
                st.success("Visualizing the results...")

                # Extracting years from original document
            years = []
            for doc in documents:
                if 'Date' in doc.page_content:
                    dates = doc.page_content.split(', ')[1:]  # Extract dates
                    years = [date.split('-')[0] for date in dates]  # Extract years

            # Ensure we only take the last 'n' years that correspond to the data
            if len(years) > len(data_matches):
                years = years[:len(data_matches)]

            # Reverse the years if needed to ensure alignment (data is from most recent to oldest)
            years = years[-len(data_matches):]  # Take the last n years

            # Sort data and years in descending order of revenue
            if data_matches:  # Ensure data_matches is not empty before processing
                sorted_data = sorted(zip(data_matches, years), reverse=True)

                # Separate data and years
                data, years = zip(*sorted_data)


                # Create a Plotly figure
                fig = go.Figure(data=[go.Bar(x=years, y=data_matches)])
                fig.update_layout(
                    title="Data Visualization",
                    xaxis_title="Year",
                    yaxis_title="Value"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("This data can't be visualized as a chart. Displaying as text:")
                st.text(result)

        except Exception as e:
            st.error(f"Error processing query: {e}")
