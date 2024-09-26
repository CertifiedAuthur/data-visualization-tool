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
from langchain_core.vectorstores import VectorStoreRetriever
import re

# Function to initialize LLM with the provided OpenAI API key
def initialize_llm(api_key):
    openai.api_key = api_key  # Set the OpenAI API key globally
    return OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=500)

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
def create_vector_index(docs):
    """Create FAISS vector index from documents"""
    embeddings = OpenAIEmbeddings()
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
Provide precise and accurate answers based solely on the information explicitly stated in the financial documents. Ensure 100% accuracy, with no assumptions or inferences.
Critical Guidelines:
1. Comprehensive Review: Exhaustively analyze the entire document to ensure all relevant data points are considered.
2. Absolute Accuracy: Only provide information explicitly stated in the documents; respond with "I don't know" if uncertain or ambiguous.
3. Latest Data Priority: Always utilize the most recent financial data available.
4. Context Validation: Verify answers directly against document content.
5. No Assumptions: Refrain from assuming or providing unsubstantiated information.
6. Cross-Referencing: Confirm every data point's accuracy through cross-referencing.
7. Clarity: Request clarification if documents lack sufficient detail or are ambiguous.
Response Format:
1. Answer the question directly and concisely.
2. Provide relevant context and supporting data.
3. Clearly indicate if information is unavailable or unclear.
Additional Considerations:
1. Be aware of document structure (e.g., tables, sections) to improve navigation.
2. Maintain consistent formatting and units (e.g., currency, percentages).
3. Prioritize accuracy over providing an answer.
Example Question Prefix:
"To ensure accuracy, please answer based solely on the provided financial documents. If unsure or lacking information, respond with 'I don't know'.
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
    vectorindex_openai = create_vector_index(docs)
    
    # User query input
    query = st.text_input("Enter your query (e.g., 'Show me revenue for Company XYZ'): ")

    if query:
        try:
            # Run the query and get the response
            qa_results = run_qa_chain(query, vectorindex_openai, llm)

            # Extract result from QA
            result = qa_results['result']

            # Adjust metric extraction to be more robust
            metric = extract_chart_title(query) 

            # Use regex to find all numeric values in the result
            data_matches = [int(x.replace(',', '').replace('.', '')) for x in result.split('was')[1].replace('and', '').split()]

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
            sorted_data = sorted(zip(data_matches, years), reverse=True)

            # Separate data and years
            data, years = zip(*sorted_data)

            # Create Plotly figure
            fig = go.Figure(data=[go.Bar(x=years, y=data, 
                                        hovertext=[f"{year}: {format(x, ',d')}" for x, year in zip(data, years)],
                                        hoverinfo='text')])

            # Customize layout
            fig.update_layout(
                title=f'{metric.capitalize()} Over Last {len(data)} Years',
                xaxis_title='Year',
                yaxis_title=f'{metric.capitalize()} (in billions)'
            )

            # Show Plotly figure
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing query: {e}")
