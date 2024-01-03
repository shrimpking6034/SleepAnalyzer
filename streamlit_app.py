import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='🦜🔗 Sleep Analyzer')
st.title('🦜🔗 Sleep Analyzer')

prefix = ("You are an expert on sleep quality analysis. The csv data contains two rows: 'Sleep Stage' and "
          "'Time [hh:mm:ss]'. The data has epoch length of 30 seconds.")
# Load CSV file
def load_csv(input_csv):
  if input_csv.type == 'txt':
    df = pd.read_csv(input_csv, skiprows=17, delimiter='\t', encoding='utf-8-sig')
    df = df[['Sleep Stage', 'Time [hh:mm:ss]']]
  else:
    df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df

# Generate LLM response
def generate_response(csv_file, input_query):
  llm = ChatOpenAI(model_name='gpt-4-0613', temperature=0, openai_api_key=openai_api_key)
  df = load_csv(csv_file)
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, prefix=prefix)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload Sleep Data', type=['csv', 'txt'])
question_list = [
  "How well did I sleep based on the distribution of sleep stages?",
  "What's the total duration of sleep?",
  'Breakdown of the sleep stages.',
  'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text is 'Other':
  query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled=not uploaded_file)
if not openai_api_key.startswith('sk-'):
  st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
  st.header('Output')
  generate_response(uploaded_file, query_text)

