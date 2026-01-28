# =========================
# 📦 IMPORTS
# =========================
import streamlit as st
import tempfile
import os

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================
# ⚙️ PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DATA2DASH – PDF to Knowledge Graph",
    layout="wide"
)

st.title("📄 PDF Knowledge Graph Extractor")


# =========================
# 🔧 SIDEBAR CONFIGURATION
# =========================
with st.sidebar:
    st.header("🔑 LLM Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")

    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Lower = more precise (recommended for knowledge graphs)"
    )

    st.divider()

    st.header("📄 PDF Processing")
    max_pages = st.slider(
        "Max PDF Pages",
        1, 20, 5,
        help="Limit pages to avoid overwhelming the LLM"
    )

    chunk_size = st.slider(
        "Chunk Size",
        500, 2000, 1000, 100,
        help="Text size per LLM call"
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0, 300, 150, 25,
        help="Overlap helps capture cross-paragraph relations"
    )

    st.divider()

    st.header("🗄️ Neo4j Connection")
    neo4j_url = st.text_input("Neo4j URL", value="bolt://localhost:7687")
    neo4j_user = st.text_input("Username", value="neo4j")
    neo4j_pass = st.text_input("Password", type="password")


# =========================
# 🧠 UTILITY FUNCTIONS
# =========================
def extract_text_from_pdf(uploaded_file, max_pages: int):
    """Extract text from uploaded PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text = " ".join(p.page_content for p in pages[:max_pages])
    os.remove(tmp_path)

    return text, len(pages)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """Split text into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def visualize_graph(graph_documents):
    """Visualize extracted knowledge graph using PyVis"""
    G = nx.DiGraph()

    color_map = {
        "AI Model": "#FF6B6B",
        "Parameter": "#4ECDC4",
        "Organization": "#1A535C",
        "Concept": "#F7B801",
        "Metric": "#9B5DE5",
        "Methodology": "#00BBF9"
    }

    for doc in graph_documents:
        for rel in doc.relationships:
            src = rel.source
            tgt = rel.target

            G.add_node(
                src.id,
                label=src.id,
                title=f"Type: {src.type}",
                color=color_map.get(src.type, "#97C2FC")
            )

            G.add_node(
                tgt.id,
                label=tgt.id,
                title=f"Type: {tgt.type}",
                color=color_map.get(tgt.type, "#97C2FC")
            )

            G.add_edge(
                src.id,
                tgt.id,
                label=rel.type,
                title=rel.type
            )

    net = Network(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )

    net.from_nx(G)
    net.toggle_physics(True)

    path = "knowledge_graph.html"
    net.save_graph(path)

    return path


# =========================
# 🖥️ UI LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Research Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    manual_text = st.text_area(
        "Or paste text manually (optional)",
        height=150
    )

    process_btn = st.button("🚀 Extract Knowledge Graph")


# =========================
# 🚀 MAIN EXECUTION LOGIC
# =========================
if process_btn:

    if not groq_api_key:
        st.error("❌ Please provide a Groq API Key.")
        st.stop()

    if not uploaded_file and not manual_text:
        st.error("❌ Upload a PDF or paste text.")
        st.stop()

    try:
        # -------- TEXT EXTRACTION --------
        if uploaded_file:
            with st.spinner("📖 Reading PDF..."):
                extracted_text, total_pages = extract_text_from_pdf(
                    uploaded_file, max_pages
                )
        else:
            extracted_text = manual_text
            total_pages = "manual"

        # -------- TEXT CHUNKING --------
        chunks = chunk_text(
            extracted_text,
            chunk_size,
            chunk_overlap
        )

        docs = [Document(page_content=c) for c in chunks]

        # -------- LLM INITIALIZATION --------
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=temperature
        )

        # -------- GRAPH TRANSFORMER --------
        allowed_nodes = [
            "AI Model", "Parameter", "Organization",
            "Concept", "Metric", "Methodology"
        ]

        allowed_relationships = [
            "DEVELOPED_BY", "USES", "IMPROVES",
            "FOUNDED_ON", "EVALUATED_BY"
        ]

        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships
        )

        # -------- GRAPH EXTRACTION --------
        with st.spinner("🧠 Extracting knowledge graph..."):
            graph_documents = transformer.convert_to_graph_documents(docs)

        # -------- SAVE TO NEO4J --------
        graph_db = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_pass
        )

        graph_db.add_graph_documents(
            graph_documents,
            include_source=True
        )

        st.success(
            f"✅ Extracted from {total_pages} pages "
            f"({len(chunks)} chunks) and saved to Neo4j!"
        )

        # -------- VISUALIZATION --------
        with col2:
            st.markdown("### 🕸️ Knowledge Graph Visualization")

            if graph_documents:
                graph_path = visualize_graph(graph_documents)
                with open(graph_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=700)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
