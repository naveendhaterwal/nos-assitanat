---
title: Nos Assistant
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: docker
app_file: main.py
pinned: false
---

# Nos Assistant Backend

An intelligent, context-aware AI backend built specifically for the Nosana Decentralized GPU Network.

---

## 🌟 Overview

Nos Assistant is an autonomous AI Copilot integrated directly into the Nosana ecosystem. It provides real-time, context-aware answers regarding GPU availability, live market pricing, and Nosana CLI deployments using advanced Retrieval-Augmented Generation (RAG) and LangGraph routing.

### Key Features
- **🧠 Ecosystem Intelligence:** Deep knowledge of Nosana's decentralized compute network.
- **⚡ Live Market Data:** Fetches live GPU pricing and availability from Nosana's market endpoints.
- **🔍 Qdrant RAG:** Vector-based documentation search for accurate deployment and hardware troubleshooting.

---

## 🚀 Getting Started

The backend is a high-performance Python application built with LangChain, LangGraph, and FastAPI.

### Local Setup

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Make sure to set your GROQ_API_KEY, QDRANT_API_KEY, QDRANT_URL, and MARKETS_API_KEY inside .env
```

#### Running Locally
```bash
python3 main.py
```
> The backend will run continuously on `http://localhost:8787`.

---

## ☁️ Deploying to the Cloud

A `Dockerfile` is provided for zero-downtime deployment to platforms like **Koyeb**, **Fly.io**, or **Render**.

```dockerfile
docker build -t nos-assistant-backend .
docker run -p 8787:8787 nos-assistant-backend
```

---

## ⚙️ Environment Variables

For the backend to function, ensure your `.env` contains:
- `GROQ_API_KEY` - Used for LLaMA 3 inference.
- `QDRANT_API_KEY` / `QDRANT_URL` - Vector database containing Nosana documentation.
- `MARKETS_API_KEY` - Authorization for live Nosana market endpoint queries.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, LangGraph, Qdrant, Uvicorn.
- **AI Models**: Groq (Llama 3) for high-speed routing and generation.
