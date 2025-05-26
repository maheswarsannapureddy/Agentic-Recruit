

## Project Context

Hiring the right talent is critical to a startup’s success, yet designing an effective, legally compliant, and tailored hiring process can be complex and time-consuming for HR teams, especially in fast-moving early-stage companies.

**SmartHire AI** is an intelligent AI-powered hiring assistant designed to simplify and automate the creation of hiring strategies for startups. Using agent-based AI and natural language understanding, it helps HR professionals:

* Generate customized job descriptions aligned with specific roles, experience levels, and company values.
* Access salary benchmark data to set competitive compensation packages based on role, location, and company stage.
* Outline structured interview plans tailored to roles and seniority levels.
* Check hiring plans for compliance with regional labor laws and regulations.
* Answer ad-hoc HR questions interactively to assist decision-making.

This system combines several advanced AI components and frameworks:

* **LangChain Agents:** To orchestrate multi-step reasoning, manage workflows, and invoke specific tools based on user intents.
* **Ollama LLM:** For natural language generation and understanding to craft job descriptions and respond to free-text HR questions.
* **Vector Search (HuggingFace Embeddings + FAISS):** To provide quick retrieval of relevant documents and contextual information during interactions.
* **Function Tools:** Custom tool functions encapsulate discrete HR tasks such as salary lookups, interview plan creation, and compliance checks, enabling modular and interpretable agent workflows.
* **Memory Buffer:** Conversation history is maintained to ensure continuity in dialogues and personalized assistance.
* **Streamlit UI:** A simple and interactive web interface where users can input company profiles, generate hiring plans, and chat with the AI assistant.

By integrating these components, SmartHire AI provides a scalable, transparent, and extensible framework that empowers HR teams to make data-driven hiring decisions while reducing administrative overhead.

