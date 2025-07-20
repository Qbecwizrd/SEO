# search_engine_with_tools.py

# 🔍 Search Engine with Tools and Agents
# 🧠 Arxiv + Wikipedia Research Integration

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 🔐 Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# 🌐 Wikipedia Tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(f"Wikipedia tool loaded: {wiki.name}")  # Should print 'wikipedia'

# 📄 Arxiv Tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(f"Arxiv tool loaded: {arxiv.name}")  # Should print 'arxiv'

# 🧰 Combine tools
tools = [arxiv, wiki]
