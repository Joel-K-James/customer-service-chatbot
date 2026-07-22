##  Project Overview

This project builds an **Intelligent Customer Service System** that automatically:
- Categorises customer inquiries (ORDER, CANCEL, INVOICE, SHIPPING, PAYMENT)
- Analyses customer sentiment (positive, negative, neutral)
- Extracts key entities (customer name, product, date, money, issue)
- Retrieves relevant company policies via RAG (FAISS vector search)
- Generates natural language responses using a LoRA fine-tuned LLM (Qwen 2.5-1.5B-Instruct)
- Escalates complex or negative cases to human agents automatically
- Displays everything through a Streamlit user interface with real-time NLP analytics
