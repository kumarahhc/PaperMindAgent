# General information

This is the second assignment in the course AI Engineering, to be completed individually.

We use the following proxy API in this course: https://5f5832nb90.execute-api.eu-central-1.amazonaws.com/v1
It redirects all requests to OpenRouter [https://openrouter.ai/].
You do not need an API key. See the provided `proxy-text.py` file for an example.

Important: the API url must be kept secret!
If you use GitHub, load the API url from an .env file. Do NOT commit this .env file to GitHub.


# Create a RAG-enabled scientific literature assistant

Create a RAG agent that can answer questions about the course literature.

First, go to the literature Excel: https://docs.google.com/spreadsheets/d/1D5OBPuy1aj7od7OCEr76brFXTEWnvQQrU2Ylq8nagcY/edit?usp=sharing
Download three papers of your choice into a local folder.
Come up with three questions for each paper, ideally relating to methods, results, and limitations.

Then implement two scripts: indexer.py and query.py

## indexer.py

The goal is to process all PDF files in the local folder and load them into a vector store.

- Use a python library of your choice to extract the text from the downloaded PDFs (recommended: pypdf).
- Use the LangChain library to split the texts into chunks. Use a splitting strategy of your choice. Choose a chunk size and optional overlap.
- Use the huggingface model "sentence-transformers/all-mpnet-base-v2" to encode the chunks into embeddings.
- Use FAISS as local vectorstore and load the embeddings into the store. [You can use FAISS-gpu, however this is harder to install. FAISS-cpu is recommended.]

## query.py

The goal is to answer questions based on the collected articles.

Use the "openai/gpt-4.1-mini" model for generating the answer.

The script must implement the router pattern.
The router receives the user's question, and hands off to the best specialised agent.

Implement these four specialised agents:
1. Methods Analyst:
    answers questions about methods/setup/dataset/metrics
2. Results Extractor:
    answers questions about results/numbers/ablations/comparisons
3. Skeptical Reviewer:
    answers questions about limitations/threats/critique
4. General Synthesizer:
    answers other questions

Each agent must cite the respective document and page number.
Output the agent's name before printing the agent's answer.

Use either the OpenAI Swarm framework (highly recommended) or LangChain to implement all five agents (router + specialised agents).
https://github.com/openai/swarm
https://docs.langchain.com/

Write a suitable system prompt for each agent, including the router agent.

The implementation must use native features of the chosen framework (e.g., not "if" conditions).

User questions may be collected interactively (in a while loop; recommended) or via command line arguments.

# Deliverables

1. Provide the scripts `indexer.py` and `query.py`.

2. Provide a `README.md` (for human consumption) with:
- Instructions on how to execute the scripts (e.g., command line parameters?)
- A list of the three filenames that you downloaded
- Your nine questions
- The agent's answers for each of the nine question (copy-paste from outputs)

You may paste a link to a GitHub repo, or upload a zip file (including indexer.py, query.py and README.md).
If using GitHub, remember to never commit the API url and .env file.

N.B.: My reference implementation has 42 lines (indexer.py) and 119 lines (query-swarm.py) or 145 lines (query-langchain.py).
If your implementation is significantly longer, you may be overengineering.
