## Features

- PDF text extraction and semantic chunking
- Vector embeddings with FAISS indexing
- Multi-agent processing pipeline:
  - Information extraction
  - Data organization
  - Query handling
- Context-aware question answering
- Support for multiple AI models (DeepSeek, OpenAI)

## System Architecture

├── Core Processing
│ ├── PDF Text Extraction
│ ├── Semantic Chunking
│ └── Vector Embedding
├── AI Processing Layer
│ ├── Information Extraction Agent
│ ├── Data Organization Agent
│ └── Query Handling Agent
├── Interface
│ └── Streamlit Web UI
└── Configuration
├── Model Management
└── API Integration

## Set API keys:
export DEEPSEEK_API_KEY="your_api_key"
export OPENAI_API_KEY="your_api_key"

## This implementation:

Combines both code files into a single streamlined application

Uses environment variables for API configuration

Implements a clean architecture pattern

Includes caching mechanisms for performance

Follows security best practices for key management

Provides comprehensive documentation

## To use:

Replace API keys with your actual credentials

Install required dependencies

Run with streamlit run main.py