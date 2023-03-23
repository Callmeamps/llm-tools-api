import streamlit as st
from langchain.utilities import RequestsWrapper

requests = RequestsWrapper()
user_prompt = st.text_area("Ask a question")
prompt_id = st.number_input("id", step=1, value=0)
if st.button("Submit", use_container_width=True):
  response = requests.post("http://127.0.0.1:8000/new_prompt/" + str(prompt_id), data={"user_prompt": user_prompt})
  st.write(response)