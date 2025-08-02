# 🧠 AI Research Paper Summarizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/AI-Powered-orange.svg" alt="AI">
</p>

> Automatically fetch, summarize, and search through the latest AI/ML research papers from ArXiv using state-of-the-art LLMs and semantic search.

## 🌟 Features

- **📚 Automatic Paper Fetching**: Retrieve the latest AI/ML papers from ArXiv
- **🤖 Multi-Model Summarization**: Support for OpenAI, Anthropic, and local models
- **🔍 Semantic Search**: RAG-powered search to find relevant papers instantly
- **💻 Dual Interface**: Both CLI and web API for maximum flexibility
- **🚀 Production Ready**: Docker support, comprehensive testing, and clean architecture
- **📊 Smart Filtering**: Filter by date, citations, authors, and keywords

## 🎯 Quick Start

### 1. Clone and Setup

```bash
git clone 
cd ai-research-summarizer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
copy .env.example .env
# Edit .env with your API keys (optional - works with local models too!)