# MyProject Crew

Welcome to the MyProject Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/my_project/config/agents.yaml` to define your agents
- Modify `src/my_project/config/tasks.yaml` to define your tasks
- Modify `src/my_project/crew.py` to add your own logic, tools and specific args
- Modify `src/my_project/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the my_project Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

# 📘 README – CrewAI Review Analysis System

## 📌 Introduction

This project uses **CrewAI + Large Language Models (LLMs)** to analyze user and item data based on reviews.

The system performs:
- **User profile analysis**
- **Item (product/service) analysis**
- Aggregation of insights for **recommendation and decision support**

The input data is provided via JSON containing:
- `user_id`
- `item_id`

---

## 🧱 Project Structure

```
my_project/
│
├── src/my_project/
│   ├── main.py        # Entry point
│   ├── crew.py        # CrewAI pipeline definition
│   └── ...
│
├── data/
│   └── test_review_subset.json
│
├── .env              # API keys
├── pyproject.toml    # Dependencies (uv)
└── README.md
```

---

## ⚙️ Setup

### 1. Create virtual environment

Recommended: Python 3.11

```
python3.11 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install dependencies

Using **uv**:

```
uv add crewai langchain langchain-community langchain-core sentence-transformers
```

Or using pip:

```
pip install crewai langchain langchain-community langchain-core sentence-transformers
```

---

### 3. Configure environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
MODEL=groq/llama-3.3-70b-versatile
```

---

## 🚀 Usage

### ▶️ Run default

```
python src/my_project/main.py
```

---

### ▶️ Run with command

```
python src/my_project/main.py run
```

---

### 🧠 Train

```
python src/my_project/main.py train 5 output.json
```

---

### 🔁 Replay

```
python src/my_project/main.py replay <task_id>
```

---

### 🧪 Test

```
python src/my_project/main.py test 5 gpt-4
```

---

### ⚡ Run with trigger payload

```
python src/my_project/main.py trigger '{"key":"value"}'
```

---

## 📥 Input Data

File:

```
data/test_review_subset.json
```

Supported formats:
- JSON array
- JSONL (one object per line)

---

## ⚙️ Workflow

1. Load the first test case  
2. Extract:
   - `user_id`
   - `item_id`  
3. Pass input into CrewAI:

```
MyProject().crew().kickoff(inputs=inputs)
```

4. LLM processes:
   - user analysis  
   - item analysis  
   - final aggregation  

---

## ⚠️ Common Issues

- `ModuleNotFoundError: langchain_community` → install langchain-community  
- `ModuleNotFoundError: sentence_transformers` → install sentence-transformers  
- Python 3.13 issues → use Python 3.11  

---

## 🧠 Notes

- Frameworks:
  - CrewAI  
  - LangChain  
  - HuggingFace Embeddings  

- Embedding model:
  BAAI/bge-small-en-v1.5

---

## 📈 Future Improvements

- Add recommendation system  
- Improve prompt engineering  
- Apply GraphRAG  
- Deploy API (FastAPI)  
- Add evaluation metrics  

---

## 👨‍💻 Author

- Vinh Vo (Võ Công Vinh)  
- Master Student – NDHU CSIE  
- NLP / LLM / Recommendation Systems

