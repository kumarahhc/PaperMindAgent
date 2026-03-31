# PaperMindAgent

A multi-agent RAG system for querying scientific papers using LangChain and FAISS as local vectorstore.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment** — create a `.env` file in the project root:
   ```
   PROXY_BASE_URL=https://<your-proxy-url>
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   LLM_MODEL=openai/gpt-4o-mini
   FAISS_PATH=vector_store.index
   TOP_K_CHUNKS=5
   ```

3. **Add PDFs** — place your PDF files in the `data/` folder.

## How to Run

### Step 1 — Index the papers
Run this once (and again whenever you add/remove PDFs):
```bash
python indexer.py
```

### Step 2 — Query the agent
```bash
python query.py
```
Type your question at the prompt and press Enter. Type `exit` or `quit` to stop.

### How it works
Each query is routed to a specialized agent:
| Route | Handles |
|-------|---------|
| `methods` | Methodology, datasets, architectures, metrics |
| `results` | Benchmark scores, comparisons, ablations |
| `limitations` | Weaknesses, future work, threats to validity |
| `general` | Everything else |


# File names

1. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
2. Learning from Online Videos at Inference Time for Computer-Use Agents
3. Agent Design Patterns for Assistance Systems in various Domains – a Survey

# Questions

1. What methodology was used to identify and compare the architectures of multi-agent assistance systems across different domains?
2. What design patterns were found in common in scheduling-based agent assistance systems compared to other domains?
3. What are the main limitations of the finding and reporting on multi-agent assistance systems?
4. How does the  Agentic Context Engineering(ACE) framework update its context over time without rewriting the entire context?
5. What performance improvements did ACE achieve compared to baseline methods on the agent and financial analysis benchmarks?
6. In what situations does the ACE framework fail to improve performance, and what causes this failure?
7. How are online tutorial videos processed and converted into structured trajectories that a computer-use agent can follow?
8. How does video demonstrations are used to improve agent's performance compared to using only text tutorials or transcripts on desktop tasks?
9. What are the main challenges or weaknesses of the proposed video-based inference-time learning approach for computer-use agents?

# Answers

1. The methodology used to identify and compare the architectures of multi-agent assistance systems across different domains involved conducting a literature review of existing multi-agent system (MAS) architectures. The results of this literature review were then analyzed to derive insights about the different architectures and to identify design patterns based on the application domain and the level of automation (Source: Agent Design Patterns for Assistance Systems .pdf, Page 1).

2. The paper identifies that in the scheduling domain, all investigated assistance system architectures utilize a dedicated Coordination-Agent to manage the complexity of scheduling at a centralized location. This contrasts with other domains, where a dedicated central Coordination-Agent is only present in some architectures. Additionally, the scheduling domain consistently employs System-Component-Agents (active) to represent the interests of the corresponding components (Source: Agent Design Patterns for Assistance Systems .pdf, Page 5).  However, it is noted that the scheduling domain does not include Database- or Aggregation-Agents in its architecture, which may differ from other domains where such agents could be present (Source: Agent Design Patterns for Assistance Systems .pdf, Page 5). Thus, the common design patterns in scheduling-based agent assistance systems include the use of a centralized Coordination-Agent and System-Component-Agents, while the absence of Database- or Aggregation-Agents is a notable characteristic of this domain.

3. The context provided does not specify any explicit limitations regarding the findings and reporting on multi-agent assistance systems. It discusses the methodology, functionalities, and classifications of assistance systems but does not detail any weaknesses, threats to validity, or critiques of the findings. Therefore, I cannot provide specific limitations based on the available information.

4. The Agentic Context Engineering (ACE) framework updates its context over time through a modular workflow that incorporates "incremental updates" and follows a "grow-and-refine principle." This design allows ACE to treat contexts as evolving playbooks that accumulate, refine, and organize strategies without the need for monolithic context rewriting. By implementing structured, incremental updates, ACE maintains contexts that are comprehensive, scalable, and interpretable throughout the adaptation process, thereby preventing context collapse and preserving detailed, domain-specific knowledge (Source: Agentic Context Engineering .pdf, Page 2).

5. On the agent use case, ACE achieved an average improvement of 14.8% over the ReAct baseline (Source: Agentic Context Engineering .pdf, Page 7). In the context of financial analysis benchmarks, ACE delivered an average performance gain of 8.6% over strong baselines (Source: Agentic Context Engineering .pdf, Page 6). Additionally, in the offline setting with ground-truth answers, ACE surpassed other methods such as ICL, MIPROv2, and GEPA by an average of 10.9% (Source: Agentic Context Engineering .pdf, Page 8). Furthermore, ACE outperformed IBM CUGA by 8.4% in TGC and 0.7% in SGC on the harder test-challenge split (Source: Agentic Context Engineering .pdf, Page 8).

6. The ACE framework may fail to improve performance in situations where the Reflector is unable to extract meaningful insights from generated traces or outcomes. If the Reflector fails in this regard, the constructed context can become "noisy or even harmful," leading to ineffective or detrimental results. This limitation is particularly pronounced in domain-specific tasks where no model can extract useful insights, which can render the resulting context ineffective (Source: Agentic Context Engineering .pdf, Page 14).

7. The processing of online tutorial videos into structured trajectories involves a multi-step approach. First, the agent generates search queries based on a given task description to retrieve relevant online video tutorials. After filtering these videos to a small set of relevant tutorials, a processing pipeline is employed to convert each filtered video into structured demonstration trajectories. This conversion is facilitated by an off-the-shelf vision language model (VLM), which infers the relevant demonstration trajectory at each step of the video. The selected trajectory is then provided in context to assist the agent in deciding the next action (Source: Learning from Online Videos at Inference Time .pdf, Page 2-3).

8. Video demonstrations improve the agent's performance on desktop tasks by providing rich observational demonstrations that include visual information, which is crucial for understanding the context of actions. The method involves several steps:\
**Video Retrieval**: The agent retrieves and filters relevant online videos that contain useful demonstrations applicable to the testing task.\
**Video Processing**: A Video Learning Model (VLM) extracts underlying actions from the video and summarizes important segments into structured demonstration trajectories, which include descriptions and sequences of actions and observations.\
**Video Application**: During inference, a two-stage selection method is employed to select the most relevant information for the agent.\
In contrast, using only text tutorials or transcripts lacks the visual context that videos provide. The experiments indicated that when the agent was tested with text-only summaries, it performed worse than when it had access to the original video content, highlighting that "text-only summaries cannot cover all useful information in the video" (Source: Learning from Online Videos at Inference Time .pdf, Page 8). Thus, the inclusion of visual information from videos significantly enhances the agent's ability to learn and perform tasks effectively.

9. The proposed video-based inference-time learning approach for computer-use agents has several challenges and weaknesses:\
**Dependence on Visual Information**: The study indicates that "text-only summaries cannot cover all useful information in the video," suggesting that the absence of visual information significantly hampers performance (Source: Learning from Online Videos at Inference Time .pdf, Page 8). This reliance on visual data may limit the approach's applicability in scenarios where visual content is not available or is of poor quality.\
**Trajectory Selection**: The performance of the agent drops significantly when the method does not include trajectory selection, as shown in the second row of Table 4 (Source: Learning from Online Videos at Inference Time .pdf, Page 8). This highlights the challenge of effectively segmenting and selecting relevant video trajectories, which is crucial for maintaining high performance.\
**Generalization to Diverse Tasks**: While the framework shows improved performance on specific benchmarks, the ability to generalize across a wide range of tasks remains uncertain. The paper does not provide extensive evidence of how well the approach performs outside the tested scenarios.\
**Complexity of Video Processing**: The multi-step process of video retrieval, processing, and application introduces complexity that may affect the efficiency and speed of the agent's learning and decision-making processes (Source: Learning from Online Videos at Inference Time .pdf, Page 3).\
These challenges suggest that while the approach shows promise, further work is needed to address these limitations and enhance the robustness and versatility of the system.


# Any additional comments (optional)


