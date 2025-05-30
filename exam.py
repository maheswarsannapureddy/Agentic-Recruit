# import subprocess
# import streamlit as st
# from typing import List, Dict, Any
# import asyncio

# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import tool as FunctionTool
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama

# # Initialize event loop for async operations
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# # Initialize Ollama LLM
# llm = Ollama(model="llama3.2", temperature=0.5)

# # Initialize embedding model and vector store
# embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# documents = [
#     Document(page_content="California employment law requires minimum $16/hour wage.", metadata={"type": "legal", "location": "California"}),
#     Document(page_content="Startup technical roles typically take 45-60 days to fill.", metadata={"type": "metrics", "role": "technical"}),
#     Document(page_content="Standard interview process: Phone screen -> Technical assessment -> Onsite -> Offer", metadata={"type": "process"})
# ]
# index = FAISS.from_documents(documents, embedding=embed_model)
# retriever = index.as_retriever()

# # Memory for conversational context
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Define LangChain tools with proper docstrings
# @FunctionTool
# def generate_job_description(role: str, experience_level: str, company_values: str) -> str:
#     """
#     Generate a detailed job description for a given role, experience level, and company values.
    
#     Args:
#         role: The job role/title to generate description for
#         experience_level: The experience level (e.g., 'entry', 'mid', 'senior')
#         company_values: Comma-separated list of company values
        
#     Returns:
#         A detailed job description including responsibilities, requirements, and benefits
#     """
#     values_list = [v.strip() for v in company_values.split(",")]
#     prompt = f"""Create a detailed job description for the role of {role} at {experience_level} level. 
#     Company values: {', '.join(values_list)}. 
#     Include Responsibilities, Requirements, and Benefits."""
#     response = llm.invoke(prompt)
#     return response

# @FunctionTool
# def get_salary_benchmark(role: str, location: str, stage: str) -> Dict[str, Any]:
#     """
#     Return simulated salary benchmark data for a role, location, and company stage.
    
#     Args:
#         role: The job role to get salary data for
#         location: The geographic location
#         stage: The company stage (e.g., 'seed', 'Series A', 'established')
        
#     Returns:
#         Dictionary with salary benchmark data including percentiles
#     """
#     salary_data = {
#         "CTO": {"50th": 150000, "75th": 180000},
#         "Product Manager": {"50th": 120000, "75th": 140000},
#         "AI Engineer": {"50th": 120000, "75th": 150000},
#     }
#     data = salary_data.get(role, {"50th": 100000, "75th": 120000})
#     return {
#         "role": role,
#         "location": location,
#         "stage": stage,
#         "50th_percentile": data["50th"],
#         "75th_percentile": data["75th"]
#     }

# @FunctionTool
# def create_interview_plan(role: str, level: str) -> Dict[str, Any]:
#     """
#     Create a standard interview process plan for a role and level.
    
#     Args:
#         role: The job role to create interview plan for
#         level: The experience level (e.g., 'junior', 'senior')
        
#     Returns:
#         Dictionary with interview process steps and estimated timeline
#     """
#     process = [
#         "Initial phone screen (30 mins)",
#         "Technical assessment (1 hour)",
#         "Team interview (2 hours)",
#         "Culture fit interview (1 hour)",
#         "Reference checks"
#     ]
#     estimated_time = "3-4 weeks"
#     return {
#         "role": role,
#         "level": level,
#         "process": process,
#         "estimated_time": estimated_time
#     }

# @FunctionTool
# def check_compliance(location: str) -> Dict[str, Any]:
#     """
#     Check legal compliance rules for a given location.
    
#     Args:
#         location: The geographic location to check compliance for
        
#     Returns:
#         Dictionary with compliance requirements and potential issues
#     """
#     compliance_rules = {
#         "California": ["Minimum wage $16/hr", "Anti-discrimination laws", "Mandatory harassment training"],
#         "New York": ["Minimum wage $15/hr", "Salary history ban", "Paid family leave"]
#     }
#     rules = compliance_rules.get(location, ["Standard US employment laws apply"])
#     return {
#         "location": location,
#         "requirements": rules,
#         "issues": []
#     }

# tools = [
#     generate_job_description,
#     get_salary_benchmark,
#     create_interview_plan,
#     check_compliance,
# ]

# # Initialize the agent with tools, LLM, and memory
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     memory=memory,
#     verbose=True
# )

# # Function to generate the complete hiring plan using the agent
# def create_hiring_plan(company_info: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Generate a comprehensive hiring plan including job descriptions, salary benchmarks,
#     interview processes, and compliance checks for all roles.
#     """
#     plan = {}
#     for role in ["CTO", "Product Manager", "AI Engineer"]:
#         job_desc = agent.run(
#             f"generate_job_description role={role} experience_level=senior company_values={','.join(company_info['values'])}"
#         )
#         salary = agent.run(
#             f"get_salary_benchmark role={role} location={company_info['location']} stage={company_info['stage']}"
#         )
#         interview = agent.run(
#             f"create_interview_plan role={role} level=senior"
#         )
#         plan[role] = {
#             "job_description": job_desc,
#             "salary_range": salary,
#             "interview_process": interview
#         }
    
#     compliance = agent.run(
#         f"check_compliance location={company_info['location']}"
#     )

#     return {
#         "company": company_info["name"],
#         "hiring_plan": plan,
#         "compliance_check": compliance
#     }


# st.set_page_config(page_title="SmartHire AI Planner", page_icon="🧠", layout="wide")
# st.title(" SmartHire  AI-Powered Hiring Assistant")
# st.write("Plan your hiring strategy and get answers to HR-related questions.")


# with st.sidebar:
#     st.header("Company Profile")
#     company_profile = {
#         "name": "TechNova",
#         "industry": "AI",
#         "team_size": 15,
#         "location": "California",
#         "stage": "Series A",
#         "values": ["innovation", "diversity", "agility"]
#     }
#     st.json(company_profile)


# st.header(" Ask HR Questions")
# st.write("Get answers to your specific HR and hiring questions.")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "user_input" not in st.session_state:
#     st.session_state.user_input = ""


# # user_input = st.text_input("Your question:", 
# #                          placeholder="E.g., What's the interview process for a Product Manager?",
# #                          key="user_input",
# #                          value=st.session_state.user_input)


# if st.session_state.get("clear_input", False):
#     st.session_state["user_input"] = ""
#     st.session_state["clear_input"] = False

# user_input = st.text_input(
#     "Your question:",
#     placeholder="E.g., What's the interview process for a Product Manager?",
#     key="user_input"
# )


# if st.button("Ask") or user_input:
#     if user_input: 
#         try:
#             with st.spinner("Thinking..."):
#                 answer = agent.run(user_input)
#             st.session_state.chat_history.append(("You", user_input))
#             st.session_state.chat_history.append(("Assistant", answer))
            
           
#             st.session_state.user_input = ""
           
#             st.rerun()
#         except Exception as e:
#             st.error(f"Error processing your question: {str(e)}")

# st.subheader("Conversation History")
# for speaker, message in st.session_state.chat_history:
#     if speaker == "You":
#         st.markdown(f"** {speaker}:** {message}")
#     else:
#         st.markdown(f"** {speaker}:** {message}")
#         st.divider()




# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#         background-color: #f8fafc;
#     }
    
#     .stButton>button {
#         background-color: #4f46e5;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#         border: none;
#         transition: all 0.2s;
#     }
    
#     .stButton>button:hover {
#         background-color: #4338ca;
#         transform: translateY(-1px);
#     }
    
#     .st-expander {
#         background-color: white;
#         border-radius: 12px;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         padding: 1rem;
#         margin-bottom: 1rem;
#     }
    
#     .stTextInput>div>div>input {
#         border-radius: 8px;
#         padding: 0.75rem;
#     }
    
#     .stAlert {
#         border-radius: 8px;
#     }
    
#     .stMarkdown {
#         line-height: 1.6;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )




import streamlit as st
import asyncio
from typing import Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool as FunctionTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Ensure proper async event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize LLM and embedding
llm = Ollama(model="llama3.2", temperature=0.5)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Define retrievable documents
documents = [
    Document(page_content="California employment law requires minimum $16/hour wage.", metadata={"type": "legal", "location": "California"}),
    Document(page_content="Startup technical roles typically take 45-60 days to fill.", metadata={"type": "metrics", "role": "technical"}),
    Document(page_content="Standard interview process: Phone screen -> Technical assessment -> Onsite -> Offer", metadata={"type": "process"})
]

# Vector store and retriever
index = FAISS.from_documents(documents, embedding=embed_model)
retriever = index.as_retriever()

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------- LangChain Tools ----------------

@FunctionTool
def generate_job_description(role: str, experience_level: str, company_values: str) -> str:
    """Generate a detailed job description."""
    values_list = [v.strip() for v in company_values.split(",")]
    prompt = f"""Create a detailed job description for the role of {role} at {experience_level} level. 
    Company values: {', '.join(values_list)}. Include Responsibilities, Requirements, and Benefits."""
    return llm.invoke(prompt)

@FunctionTool
def get_salary_benchmark(role: str, location: str, stage: str) -> Dict[str, Any]:
    """Return salary benchmark data."""
    salary_data = {
        "CTO": {"50th": 150000, "75th": 180000},
        "Product Manager": {"50th": 120000, "75th": 140000},
        "AI Engineer": {"50th": 120000, "75th": 150000},
    }
    data = salary_data.get(role, {"50th": 100000, "75th": 120000})
    return {
        "role": role,
        "location": location,
        "stage": stage,
        "50th_percentile": data["50th"],
        "75th_percentile": data["75th"]
    }

@FunctionTool
def create_interview_plan(role: str, level: str) -> Dict[str, Any]:
    """Create a standard interview process plan."""
    process = [
        "Initial phone screen (30 mins)",
        "Technical assessment (1 hour)",
        "Team interview (2 hours)",
        "Culture fit interview (1 hour)",
        "Reference checks"
    ]
    return {
        "role": role,
        "level": level,
        "process": process,
        "estimated_time": "3-4 weeks"
    }

@FunctionTool
def check_compliance(location: str) -> Dict[str, Any]:
    """Check legal compliance for a location."""
    compliance_rules = {
        "California": ["Minimum wage $16/hr", "Anti-discrimination laws", "Mandatory harassment training"],
        "New York": ["Minimum wage $15/hr", "Salary history ban", "Paid family leave"]
    }
    rules = compliance_rules.get(location, ["Standard US employment laws apply"])
    return {"location": location, "requirements": rules, "issues": []}

@FunctionTool
def hr_search_tool(query: str) -> str:
    """Search internal HR knowledge base using semantic retrieval."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"- {doc.page_content}" for doc in docs])

# Register tools
tools = [
    generate_job_description,
    get_salary_benchmark,
    create_interview_plan,
    check_compliance,
    hr_search_tool
]

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

#  Streamlit UI 

st.set_page_config(page_title="SmartHire AI Planner", page_icon="", layout="wide")
st.title(" SmartHire  AI-Powered Hiring Assistant")
st.write("Plan your hiring strategy and get answers to HR-related questions.")

# Sidebar: Company profile
with st.sidebar:
    st.header("Company Profile")
    company_profile = {
        "name": "TechNova",
        "industry": "AI",
        "team_size": 15,
        "location": "California",
        "stage": "Series A",
        "values": ["innovation", "diversity", "agility"]
    }
    st.json(company_profile)

# Input
st.header(" Ask HR Questions")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if st.session_state.get("clear_input", False):
    st.session_state["user_input"] = ""
    st.session_state["clear_input"] = False

user_input = st.text_input(
    "Your question:",
    placeholder="E.g., What's the interview process for a Product Manager?",
    key="user_input"
)

if st.button("Ask") or user_input:
    if user_input: 
        try:
            with st.spinner("Thinking..."):
                answer = agent.run(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Assistant", answer))
            st.session_state.user_input = ""
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Chat history
st.subheader("Conversation History")
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
    st.divider()

# Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    .st-expander {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 0.75rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
