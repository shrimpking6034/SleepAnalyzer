import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='🦜🔗 Grocery GPT')
st.title('🦜🔗 Grocery GPT')

# prefix = ("You are an expert on sleep quality analysis. The csv data contains two rows: 'Sleep Stage' and "
#           "'Time [hh:mm:ss]'. The data has epoch length of 30 seconds.")
prefix = ("You are a helpful asistant for housewives on grocery store. Share the best deals based on the query.")
# Load CSV file
def load_csv(input_csv):
  # print(input_csv.name, input_csv.type)
  df = pd.read_csv(input_csv, skiprows=17, delimiter='\t', encoding='EUC-KR')
  df = df[['Sleep Stage', 'Time [hh:mm:ss]']]
  # df = pd.read_csv(input_csv, encoding='EUC-KR')
  with st.expander('See DataFrame'):
    st.write(df)
  return df

# grocery
def load_csv2(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df
# Generate LLM response
def generate_response(csv_file, input_query):
  llm = ChatOpenAI(model_name='gpt-4-0613', temperature=0, openai_api_key=openai_api_key)
  tmp = []
  for f in csv_file:
    tmp += [load_csv2(f)]
  df = pd.concat(tmp)
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, prefix=prefix)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload Sleep Data', type=['csv', 'txt'], accept_multiple_files=True)
# question_list = [
#   "How well did I sleep according to the distribution of Sleep Stage?",
#   "What's the total duration of sleep?",
#   'Breakdown of the sleep stages.',
#   'Other']

question_list = [
  "What are the best deals of fruits?",
  "What are the best deals of meat?",
  'Are there mangos with discount?',
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
