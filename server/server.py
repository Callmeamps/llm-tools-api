from langchain import LLMChain, PromptTemplate, Cohere, HuggingFaceHub, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, SelfAskWithSearchChain
#from langchain.tools import BaseTool
from pydantic import BaseModel
import sys
sys.path.insert(1, '/workspaces/llm-tools-api')
import config
#from langchain.chains import LLMRequestsChain, LLMChain
#from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from fastapi import FastAPI

app = FastAPI()

search = SerpAPIWrapper()
chatgpt = ChatOpenAI(temperature=0)
search_openai = SelfAskWithSearchChain(llm=chatgpt, search_chain=search, verbose=True)

command_xl = Cohere(temperature=0, model="command-xlarge-20221108")
search_cohere = SelfAskWithSearchChain(llm=command_xl, search_chain=search, verbose=True)

flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":1})
search_huggingfacehub = SelfAskWithSearchChain(llm=flan_ul2, search_chain=search, verbose=True)

chains = [search_openai, search_cohere, search_huggingfacehub]

tools = [
    Tool(
        name = "Search with ChatGPT",
        func=search_openai.run,
        description="Powerful chat based LLM, useful for when you need to answer complex or nuanced questions."
    ),
    Tool(
        name="Search with Command XL",
        func=search_cohere.run,
        description="Powerful command based LLM, useful for when you need to answer general questions"
    ),
    Tool(
        name = "Search with Flan-UL2",
        func=search_huggingfacehub.run,
        description="Chat based LLM, useful for when you need to answer questions about current events"
    )
]

class UserQuestion(BaseModel):
  user_prompt: str

class AIReply(BaseModel):
  reply: str = None


chatgpt_agent = initialize_agent(tools, chatgpt, agent="zero-shot-react-description", verbose=False)

respones = {}

@app.post("/new_prompt/{prompt_id}")
async def search_w_openai(prompt_id: int, user_prompt: UserQuestion):
  if prompt_id in respones:
    return {"Error": f"Prompt {prompt_id} already exists"}
  respones[prompt_id] = user_prompt
  #respones[reply] = chatgpt_agent.run(respones[prompt_id])
  #respones.append({"user_prompt": respones[prompt_id]})
  return respones[prompt_id]

@app.get("/get_all")
async def get_all_prompts():
  return respones