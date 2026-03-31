import os
import dotenv

from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
#Config
dotenv.load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
FAISS_PATH = os.getenv("FAISS_PATH", "vector_store.index")
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 5))
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "proxy_base_url_not_set")
# Prompts
PROMPT_ROUTER_SYSTEM = """
You are PaperMindAgent's routing agent. Your sole task it to classify the user's requests into one of the following categories:
1. "methods" - if the user's request is about methodology, experimental setup, datasets, architectures, metrics, algorithsms, or any other aspect related to the methods used in the paper.
2. "results" - if the user's request is about the results, findings, numbers, ablations, benchmark scores, performance comparisons, or tables of results.
3. "limitations" - if the user's request is about the limitations, weaknesses, threats to validity, critique, assumptions, or future work.
4. "general" - if the user's request does not fit into any of the above categories.

When classifying the user's request, you should only output one of the above categories as a single word in lowecase, 
without any explanation or additional text.
"""
PROMPT_METHODS = """
You are PaperMindAgent's methods agent. You are a specialized AI research assistant designed to answer questions about the methods used in a scientific paper.
Your task is to answer the user's question about the methods used in the paper, 

Your expertise: research methodology, experimental setup, datasets used, system architectures, evaluation metrics, and algorithmic design choices.

Instructions:
- Answer using ONLY the context provided below.
- Cite every claim with the source and including page number in the format:
    (Source: <filename>, Page <N>)
- Be precise, academic and technical. Quote key terms directly from the text when helpful.
- If the context does not contain enough information to answer, express it clearly.
  """
PROMPT_RESULTS = """
You are PaperMindAgent's results agent. You are a specialized AI research assistant designed to answer questions about the results presented in a scientific paper.
Your task is to answer the user's question about the results presented in the paper, 

Your expertise: experimental results, benchmark scores, numerical metrics, ablation studies, and performance comparisons between systems.

Instructions:
- Answer using ONLY the context provided below.
- Cite every claim with the source and including page number in the format:
    (Source: <filename>, Page <N>)
- Be precise, academic and technical. Quote key terms directly from the text when helpful.
- Always extract and report specific numbers, percentages, and metric values.
- When comparing systems, clearly state which system is better and by how much.
- If the context does not contain enough information to answer, express it clearly.
  """
  
PROMPT_LIMITATIONS = """
You are PaperMindAgent's limitations agent. You are a specialized AI research assistant designed to answer questions about the limitations, weaknesses, and future work of a scientific paper.
Your task is to answer the user's question about the limitations of the paper, 

Your expertise: limitations, weaknesses, threats to validity, critique, assumptions, and future work mentioned in the paper.

Instructions:
- Answer using ONLY the context provided below.
- Cite every claim with the source and including page number in the format:
    (Source: <filename>, Page <N>)
- Be precise, academic and technical. Quote key terms directly from the text when helpful.
- Always extract and report specific numbers, percentages, and metric values.
- When comparing systems, clearly state which system is better and by how much.
- If the context does not contain enough information to answer, express it clearly.
  """

PROMPT_GENERAL = """
You are PaperMindAgent's general agent. You are a specialized AI research assistant designed to answer general questions about a scientific paper.
Your task is to answer the user's question about the paper in general, 

Your expertise: limitations, weaknesses, threats to validity, critique, assumptions, and future work mentioned in the paper.

Instructions:
- Answer using ONLY the context provided below.
- Cite every claim with the source and including page number in the format:
    (Source: <filename>, Page <N>)
- Be precise, academic and technical. Quote key terms directly from the text when helpful.
- Always extract and report specific numbers, percentages, and metric values.
- When comparing systems, clearly state which system is better and by how much.
- If the context does not contain enough information to answer, express it clearly.
  """
LLM = ChatOpenAI(model=LLM_MODEL,temperature=0,api_key=SecretStr("no-key-required"),base_url=PROXY_BASE_URL,max_retries=3)

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        )
    vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def format_doc(docs: list[Document]) -> str:
    doc_parts = []
    for doc in docs:
        doc_parts.append(f"(Source: {doc.metadata.get('source','Unknown')}, Page {doc.metadata.get('page','Unknown')})\n")
        doc_parts.append(doc.page_content)
    return "\n" + "-"*40 + "\n".join(doc_parts)

def make_agent(retriever, system_promt: str):# Create a RAG agent for a specific role (methods, results, limitations, general)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_promt+"\n\nContext:\n{context}"),
        ("human", "{query}"),
    ])
    agent=(
      {
        "context": itemgetter("query") | retriever | format_doc,
        "query": itemgetter("query"),
      }
      | prompt
      | LLM
      | StrOutputParser()
    )
    return agent

router_prompt = ChatPromptTemplate.from_messages([("system", PROMPT_ROUTER_SYSTEM), ("human", "{question}")])

retriever=load_vector_store().as_retriever(search_kwargs={"k": TOP_K_CHUNKS})
router_agent = router_prompt | LLM | StrOutputParser()
# Define the specialized agents for each category
AGENTS:dict[str,tuple[str,object]] = {
    "methods": ("methods analyst",make_agent(retriever, PROMPT_METHODS)),
    "results": ("results analyst",make_agent(retriever, PROMPT_RESULTS)),
    "limitations": ("limitations analyst",make_agent(retriever, PROMPT_LIMITATIONS)),
    "general": ("general analyst",make_agent(retriever, PROMPT_GENERAL)),
}

#run the paperMingAgent
def query_paperMindAgent(query: str) -> None:
    #Step 1: select router
    router_agent_name= router_agent.invoke({"question": query}).strip().lower()
    #Step 2: select agent
    agent_name, agent = AGENTS.get(router_agent_name, ("general analyst", AGENTS["general"][1]))
    #Step 3: run agent
    answer = agent.invoke({"query": query})
    print(f"\033[34mPaperMind Answer[Agent: {agent_name}]:\n \033[0m{answer}\n")
    
def print_header():
    print("\033[34m" + "-"*80  + "\n" + f"Welcome to PaperMindAgent ver 1.0, A scientific research assistant\n"+f"Framework: LangChain | Model: {LLM_MODEL}\n" + "-"*80 + "\033[0m")
    print("\033[34m" + "Type your query and press Enter.\nType exit to quit to stop.\n"+"="*40 + "\033[0m")

if __name__ == "__main__":
  print_header()
  if not os.path.exists(FAISS_PATH) or not PROXY_BASE_URL:
      print("\033[31mError: Vector store not found or PROXY_BASE_URL not set. Please run indexer.py to index the papers first and set the PROXY_BASE_URL in .env file.\033[0m")
      exit(1)
  while True:
      try:
        query = input("\033[32mYour Query: \033[0m").strip()
      except KeyboardInterrupt:
          print("\nGoodbye!")
          break
      if not query:
          continue
      if query.lower() in ("exit", "quit"):
          print("Goodbye!")
          break
      else:
          query_paperMindAgent(query)
