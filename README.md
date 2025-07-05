# RAG DeepEval Project

A comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems using DeepEval metrics to assess transformer-based Q&A models.

## 🔍 Overview

This project implements a robust evaluation system for RAG (Retrieval-Augmented Generation) models using DeepEval's custom evaluation framework. It specifically focuses on assessing transformer-based Q&A systems performance across multiple dimensions including relevance, context utilization, and completeness.

---

## 🚀 Installation

### Step 1: Install Ollama

#### On Windows:
```bash
winget install Ollama.Ollam
```
Or you can download it from https://ollama.com/download/windows


### Step 2: Pull Mistral Model
ollama pull mistral
ollama list  # Verify
ollama run mistral "Hello, how are you?"  # Test

### Step 3: Pull Mistral Model
git clone https://github.com/DBhardwaj21/RAG_Deepeval-.git
cd rag-deepeval-project

### Step 4: Create Virtual Environment and Install Dependencies
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


### Step 4: Run the python script 
pyhton Rag_Deepeval.py







