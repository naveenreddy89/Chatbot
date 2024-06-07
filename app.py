import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import SmartDataframe

API_KEY = st.secrets("API_KEY")

llm = OpenAI(api_token= API_KEY)

def chat_with_csv(df, query):
    pandas_ai = SmartDataframe(df, config={"llm":llm})
    result = pandas_ai.chat(query)
    return result

st.title("Chatbot")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv:
    data = pd.read_csv(input_csv)
    st.dataframe(data.head(5), use_container_width=True)
    
    prompt = st.text_input("Enter the query", label_visibility='collapsed', placeholder='Message Chatbot')
    
    if prompt:
        with st.spinner("Generating response..."):
            result = chat_with_csv(data, prompt)
            st.success(result)
    else:
        st.warning("Please enter a prompt")
