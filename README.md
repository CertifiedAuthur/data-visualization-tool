### Economic Data Visualization Project

##### Overview
This project is a Streamlit-based web application that allows users to visualize economic data interactively. The app leverages OpenAI embeddings, FAISS indexing, and Plotly to retrieve relevant information from a knowledge base, answer user queries, and display results as interactive charts. The focus is on providing clear, data-driven visualizations based on structured economic data.

##### Features
📊 Interactive Chart Generation: Automatically creates visual charts based on user queries using Plotly.
🔍 Natural Language Understanding: Uses OpenAI embeddings to process and understand user questions.
🔗 Efficient Document Retrieval: Utilizes FAISS (Facebook AI Similarity Search) to perform fast and accurate information retrieval.
📈 Visualization: Generates line plots, bar charts, and more to visualize economic metrics like GDP, inflation rate, and more using Matplotlib and Plotly.
🛠️ Customizability: Easily adaptable to work with various economic datasets.

##### How It Works
User Input: The user enters a query about a particular economic metric (e.g., "Show me the GDP growth rate from 2000 to 2020").
Data Retrieval: The query is processed using an LLM (Large Language Model) and matched against the available indexed documents using FAISS.
Data Analysis: The relevant data is extracted and passed to a data visualization module.
Chart Display: An interactive chart is generated and displayed to the user using Plotly and Matplotlib.

##### Setup & Installation

###### Prerequisites
Python 3.8 or higher
Streamlit
Pandas
Plotly
LangChain
OpenAI API
FAISS (langchain-community module for FAISS integration)
Matplotlib
Installation Steps
Clone the Repository:

bash
Copy code
git clone https://github.com/CertifiedAuthur/economic-data-visualization.git
cd economic-data-visualization
Create a Virtual Environment:

bash
Copy code
python -m venv venv
Activate the Virtual Environment:

On Windows:

bash
Copy code
venv\Scripts\activate
On macOS/Linux:

bash
Copy code
source venv/bin/activate
Install the Required Packages:

Install the dependencies specified in requirements.txt:

bash
Copy code
pip install -r requirements.txt
requirements.txt:

Copy code
streamlit 
pandas 
faiss-cpu 
matplotlib 
langchain 
langchain_openai
langchain_community
openpyxl
plotly
Set Up the Environment Variables:

Create a .env file in the root directory and add the following keys:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
Run the Application:

bash
Copy code
streamlit run visual.py
Open your web browser and go to http://localhost:8501 to access the application.

##### File Structure
bash
Copy code
├── README.md
├── requirements.txt
├── visual.py                # Main application script
├── data
│   └── economic_data.csv    # Sample economic dataset
└── venv                     # Virtual environment

##### Usage
Run the application by executing streamlit run visual.py.
Enter a query in the text input box, such as "Show the GDP growth from 2000 to 2018".
The system will process your query and return a chart displaying the relevant economic data using Plotly or Matplotlib for interactivity.

##### Troubleshooting
ValueError: DataFrame constructor not properly called!
This error usually indicates a problem with how data is being passed or formatted in Pandas. Make sure the data retrieval and processing steps are outputting correctly formatted dictionaries or dataframes.

TypeError: 'PromptTemplate' object cannot be converted to 'PyString'
This error occurs when a wrong type is passed to the embed_query function. Check that the input is a string and not a PromptTemplate object.

##### Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue if you have suggestions or improvements.

##### License
This project is licensed under the MIT License. See the LICENSE file for more details.

##### Contact
For any questions or support, please feel free to reach out:

Email: chibuzorauthur@gmail.com
GitHub: @CertifiedAuthur
