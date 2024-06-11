import streamlit as st
import pandas as pd
import os
import glob
from pandasai.llm import OpenAI
from pandasai import SmartDataframe

# API_KEY = st.secrets["OPENAI_API_KEY"]

llm = OpenAI(api_token="")

def clear_charts_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        os.remove(file)

def chat_with_csv(df, query):
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(query)
    return result

def display_charts(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*'))
    for image_file in image_files:
        st.image(image_file)

st.title("Chatbot")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv:
    data = pd.read_csv(input_csv)
    st.dataframe(data.head(5), use_container_width=True)

    tab1, tab2 = st.tabs(["Text Answer", "Graphical Answer"])

    with tab1:
        st.header("Text-Based Answer")

        prompt = st.text_input("Enter the query for text answer", label_visibility='collapsed', placeholder='Message Chatbot')

        if prompt:
            with st.spinner("Generating response..."):
                result = chat_with_csv(data, prompt)
                st.success(result)
        else:
            st.warning("Please enter a prompt")

    with tab2:
        st.header("Graphical Answer")

        prompt = st.text_input("Enter the query for graphical answer", label_visibility='collapsed', placeholder='Message Chatbot')

        if prompt:
            with st.spinner("Generating response..."):
                charts_folder = "exports/charts"
                clear_charts_folder(charts_folder)
                
                result = chat_with_csv(data, prompt)
                
                display_charts(charts_folder)
        else:
            st.warning("Please enter a prompt")
