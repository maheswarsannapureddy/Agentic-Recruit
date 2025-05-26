
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.tools import Tool  
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.tools import tool as FunctionTool  
import os
import json
import subprocess
import streamlit as st
from typing import List, Dict



llm = Ollama(model="llama3.2", temperature=0.5)

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


documents = [
    Document(page_content="California employment law requires minimum $16/hour wage.", metadata={"type": "legal", "location": "California"}),
    Document(page_content="Startup technical roles typically take 45-60 days to fill.", metadata={"type": "metrics", "role": "technical"}),
    Document(page_content="Standard interview process: Phone screen -> Technical assessment -> Onsite -> Offer", metadata={"type": "process"})
]
index = FAISS.from_documents(documents, embedding=embed_model)
retriever = index.as_retriever()
memory = ConversationBufferMemory(token_limit=1000)

from langchain.tools import tool

@tool
def generate_job_description(role: str, experience_level: str, company_values: List[str]) -> str:
    """Generate a job description for a given role and experience level with company values."""
    prompt = f"""Create a job description for: {role} ({experience_level} level)
    Company values: {', '.join(company_values)}
    Include: Responsibilities, Requirements, and Benefits sections"""
    return str(llm.invoke(prompt).content)

@tool
def get_salary_benchmark(role: str, location: str, stage: str) -> Dict:
    """Fetch salary benchmark data (50th and 75th percentile) for the given role and location."""
    salary_data = {
        "CTO": {"50th": 150000, "75th": 180000},
        "Product Manager": {"50th": 120000, "75th": 140000}
    }
    return {
        "role": role,
        "location": location,
        "data": salary_data.get(role, {"50th": 100000, "75th": 120000})
    }

@tool
def create_interview_plan(role: str, level: str) -> Dict:
    """Create an interview process plan for the given role and level."""
    return {
        "role": role,
        "level": level,
        "process": [
            "Initial phone screen (30 mins)",
            "Technical assessment (1 hour)",
            "Team interview (2 hours)",
            "Culture fit interview (1 hour)",
            "Reference checks"
        ],
        "estimated_time": "3-4 weeks"
    }

@tool
def check_compliance(plan: Dict, location: str) -> Dict:
    """Check legal compliance of the hiring plan in the given location."""
    compliance_rules = {
        "California": ["Minimum wage $16/hr", "Anti-discrimination laws", "Mandatory harassment training"],
        "New York": ["Minimum wage $15/hr", "Salary history ban", "Paid family leave"]
    }
    return {
        "location": location,
        "requirements": compliance_rules.get(location, ["Standard US employment laws apply"]),
        "issues": []
    }



tools = [
    FunctionTool("generate_jd", generate_job_description),
    FunctionTool("get_salary", get_salary_benchmark),
    FunctionTool("create_interview", create_interview_plan),
    FunctionTool("check_compliance", check_compliance),
]


agent = initialize_agent(
    tools=tools,  
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

# def create_hiring_plan(company_info: Dict):
#     needs_analysis = agent.invoke(
#         f"Analyze hiring needs for {company_info['name']} "
#         f"in {company_info['industry']} with {company_info['team_size']} employees"
#     )

#     role_priority = agent.invoke(
#         "Based on this startup context, recommend role hiring priority:\n"
#         f"{str(needs_analysis)}\nConsider their funding stage: {company_info['stage']}"
#     )

#     plan = {}
#     for role in ["CTO", "Product Manager"]:
#         plan[role] = {
#             "job_description": generate_job_description(role, "senior", company_info["values"]),
#             "salary_range": get_salary_benchmark(role, company_info["location"], company_info["stage"]),
#             "interview_process": create_interview_plan(role, "senior")
#         }

#     compliance = check_compliance(plan, company_info["location"])
#     return {
#         "company": company_info["name"],
#         "hiring_priority": str(role_priority),
#         "detailed_plan": plan,
#         "compliance_check": compliance
#     }

# def ollama_run(prompt: str) -> str:
#     """Run an Ollama model via CLI and get the output."""
#     result = subprocess.run(
#         ['ollama', 'run', 'llama3.2', prompt],  
#         capture_output=True,
#         text=True
#     )
#     if result.returncode != 0:
#         raise RuntimeError(f"Ollama run failed: {result.stderr}")
#     return result.stdout.strip()

# if __name__ == "__main__":
#     company_profile = {
#         "name": "TechNova",
#         "industry": "AI",
#         "team_size": 15,
#         "location": "California",
#         "stage": "Series A",
#         "values": ["innovation", "diversity", "agility"]
#     }

#     print("Generating hiring plan...")
#     try:
#         plan = create_hiring_plan(company_profile)
#         print("\nGenerated Hiring Plan:")
#         print(json.dumps(plan, indent=2))
#     except Exception as e:
#         print(f"Error generating plan: {e}")

#     print("\nAsk HR questions (type 'exit' to quit):")
#     while True:
#         user_input = input("\nHR Question: ")
#         if user_input.lower() == 'exit':
#             break
#         try:
#             response = ollama_run(user_input)
#             print(f"\nAssistant: {response}")
#         except Exception as e:
#             print(f"Error processing question: {e}")



# Define the Ollama CLI runner
def ollama_run(prompt: str) -> str:
    """Run an Ollama model via CLI and return the output."""
    result = subprocess.run(
        ['ollama', 'run', 'llama3.2', prompt],  
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama run failed: {result.stderr}")
    return result.stdout.strip()


def create_hiring_plan(profile: dict) -> dict:
    return {
        "company": profile["name"],
        "location": profile["location"],
        "hiring_plan": [
            {
                "role": "AI Engineer",
                "job_description": "Build and deploy ML models...",
                "interview_stages": ["Phone screen", "Tech interview", "Culture fit"],
                "salary_range": "$120K - $150K"
            },
            {
                "role": "Product Manager",
                "job_description": "Drive product roadmap...",
                "interview_stages": ["Intro call", "PM challenge", "Executive review"],
                "salary_range": "$100K - $130K"
            }
        ]
    }


company_profile = {
    "name": "TechNova",
    "industry": "AI",
    "team_size": 15,
    "location": "California",
    "stage": "Series A",
    "values": ["innovation", "diversity", "agility"]
}


st.set_page_config(page_title="SmartHire AI Planner", page_icon="🧠")
st.title(" SmartHire – AI-Powered Hiring Assistant")
st.write("Plan your hiring strategy and ask HR-related questions.")




with st.sidebar:
    st.header(" Company Profile")
    st.json(company_profile)

if st.button("Generate Hiring Plan"):
    with st.spinner("Generating hiring plan using LangGraph logic..."):
        try:
            plan = create_hiring_plan(company_profile)
            st.success("Hiring plan generated!")
            st.subheader("Hiring Plan")
            for role in plan["hiring_plan"]:
                st.markdown(f"###  {role['role']}")
                st.write("**Description:**", role["job_description"])
                st.write("**Interview Stages:**", ", ".join(role["interview_stages"]))
                st.write("**Salary Range:**", role["salary_range"])
        except Exception as e:
            st.error(f"Failed to generate hiring plan: {e}")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #f7f9fb;
        color: #222222;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    .stSidebar {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Ask HR-related Questions")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Your HR Question:", placeholder="e.g., What’s a good salary for a senior ML engineer?")

if user_question:
    try:
        with st.spinner("Thinking with Ollama..."):
            assistant_reply = ollama_run(user_question)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", assistant_reply))
    except Exception as e:
        st.error(f"Error: {e}")


for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
