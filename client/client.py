import streamlit as st
from langchain.utilities import RequestsWrapper

requests = RequestsWrapper()
user_prompt = st.text_area("Ask a question")
prompt_id = st.number_input("id", step=1, value=0)
request_url = "https://callmeamps-organic-acorn-wjwxx4jp7r737j-8000.preview.app.github.dev"
if st.button("Submit", use_container_width=True):
  response = requests.post(request_url + "/new_prompt/" + str(prompt_id), data={"user_prompt": user_prompt})
  st.write(response)