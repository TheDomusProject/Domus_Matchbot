
# Domus MatchBot

Domus MatchBot is an AI-powered matching tool designed to connect households in need of temporary accommodation with suitable available properties.  
The system leverages structured feature codes, vector search, and LLM-based explanations to streamline matching and improve housing allocation efficiency.

## About the Project

The project was developed as a collaborative initiative between the **University of East London (UEL)** and the **London Borough of Newham (LBN)** to address the challenges of temporary accommodation placement.  
By combining **data science, AI, and public sector expertise**, Domus MatchBot enables faster, more accurate matching while ensuring equity in allocation based on household needs, affordability, and location preferences.

This repository contains:
- **Core application code** (`.py` files) for data processing, vector database creation, and matching logic.
- **Jupyter Notebook(s)** (`.ipynb`) for **data preprocessing, exploratory analysis, and prototyping**.
- **Configuration files** and supporting scripts for deployment.

---

## Pipeline

The Domus MatchBot workflow follows these steps:

```text
1. Data Ingestion
   - Load household, property, postcode, and LHA datasets.

2. Data Preprocessing
   - Clean datasets, generate feature codes, and calculate affordability.

3. Vector Database Creation
   - Generate embeddings for households and properties using SentenceTransformers.
   - Store embeddings in FAISS indexes.

4. Matching Logic
   - Apply rule-based filters (bedroom need, affordability, zone match).
   - Perform semantic search for closest matches.

5. LLM Explanation (Optional)
   - Generate human-readable match reasons using a local LLM.

6. Streamlit Application
   - Interactive search by Reference ID, Feature Code, or Free Text prompt.
   - Export results (CSV/PDF) and email matched properties.

## Installation
Clone the repository and install dependencies:

git clone git@github.com:TheDomusProject/Domus_Matchbot.git
cd Domus_Matchbot
pip install -r requirements.txt

##Usage

Run the Streamlit app locally:
streamlit run app.py

##Dependencies

Main dependencies include:

Python 3.10+
pandas
numpy
sentence-transformers
faiss-cpu
streamlit
scikit-learn
openpyxl

Full list available in requirements.txt.

##Contributing

This is an organizational collaborative repository.
Please follow the established branching and PR guidelines when contributing.

##Notes

Sensitive data, .streamlit/secrets.toml, local datasets (.csv, .xlsx), and vector databases are excluded via .gitignore and must not be committed.

Jupyter Notebooks (.ipynb) in this repository are intended for data preprocessing, exploratory analysis, and experimentation before finalizing production-ready code.


