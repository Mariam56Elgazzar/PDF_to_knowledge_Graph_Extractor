

# 📄 PDF to Knowledge Graph Extractor

A Streamlit application that converts research papers (PDF or pasted text) into a structured **Knowledge Graph** using Large Language Models (LLMs) and stores the result in **Neo4j**, with interactive visualization.

---

## 🚀 Features

* Upload a **PDF research paper** or paste text manually
* Automatically extract:

  * Entities (AI Models, Organizations, Concepts, Metrics, etc.)
  * Relationships between entities
* Build a **Knowledge Graph** using an LLM
* Save the graph into **Neo4j**
* Interactive visualization using **PyVis**
* Performance-safe design with chunking and limits

---

## 🧠 Architecture Overview

```
PDF / Text
   ↓
Text Chunking
   ↓
LLM (Groq – LLaMA 3.3)
   ↓
Graph Extraction (Entities + Relations)
   ↓
Neo4j Graph Database
   ↓
Interactive Visualization (PyVis)
```

---

## 📦 Tech Stack

* **Frontend**: Streamlit
* **LLM**: Groq (LLaMA 3.3 – 70B)
* **Graph Extraction**: LangChain `LLMGraphTransformer`
* **Graph Database**: Neo4j
* **Visualization**: NetworkX + PyVis
* **PDF Parsing**: LangChain `PyPDFLoader`

---

## ⚙️ Configuration (Sidebar)

### 🔑 LLM Settings

* **Groq API Key** – required
* **Temperature**

  * `0.0` recommended for accurate knowledge graphs
  * Higher values may introduce hallucinated relations

### 📄 PDF Processing

* **Max PDF Pages**
  Limits the number of pages sent to the LLM for performance and cost control
* **Chunk Size**
  Controls text size per LLM call
* **Chunk Overlap**
  Helps preserve relationships across paragraph boundaries

### 🗄️ Neo4j Connection

* Neo4j URL
* Username
* Password

---

## 🧩 Supported Graph Schema

### Allowed Node Types

* `AI Model`
* `Parameter`
* `Organization`
* `Concept`
* `Metric`
* `Methodology`

### Allowed Relationship Types

* `DEVELOPED_BY`
* `USES`
* `IMPROVES`
* `FOUNDED_ON`
* `EVALUATED_BY`

This schema restriction prevents noisy or inconsistent graphs.

---

## 🛡️ Performance Design

This project is intentionally **limit-driven** to avoid:

* LLM overload
* Excessive Neo4j writes
* UI freezes during visualization

Key safeguards:

* Page limits for PDFs
* Controlled chunk size and overlap
* Restricted node and relationship types
* Directed graph visualization

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install streamlit langchain langchain-groq neo4j pyvis networkx
```

### 2️⃣ Start Neo4j

Make sure Neo4j is running (local or AuraDB).

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Enter your **Groq API Key**
2. Upload a PDF or paste text
3. Adjust chunking and page limits if needed
4. Click **Extract Knowledge Graph**
5. View the graph and find it saved in Neo4j

---

## 📊 Output

* Interactive graph visualization inside Streamlit
* Persistent Knowledge Graph stored in Neo4j
* Source text attached to graph nodes for traceability

---

## 🔮 Future Improvements

* Graph sampling for large documents
* Query-based visualization (Cypher + LIMIT)
* Ontology alignment
* Export graph as JSON / RDF
* Multi-document graph merging

---

## 📜 License

This project is intended for **research and educational purposes**.


