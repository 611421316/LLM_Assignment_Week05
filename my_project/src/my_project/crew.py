import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import JSONSearchTool
from langchain_community.embeddings import HuggingFaceEmbeddings

# Required workaround for CrewAI tools
os.environ["OPENAI_API_KEY"] = "NA"

# HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# RAG config for JSONSearchTool
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# YAML config files
agents_config = os.path.join(CONFIG_DIR, "agents.yaml")
tasks_config = os.path.join(CONFIG_DIR, "tasks.yaml")

# Data files
user_json_path = os.path.join(DATA_DIR, "user_subset.json")
item_json_path = os.path.join(DATA_DIR, "item_subset.json")
review_json_path = os.path.join(DATA_DIR, "review_subset.json")

# RAG tools
user_rag_tool = JSONSearchTool(
    json_path=user_json_path,
    collection_name="v3_hf_user_data",
    config=rag_config
)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = (
    "Search user data. IMPORTANT: input must be "
    '{"search_query": "user_id: <USER_ID> average_stars review_count yelping_since useful funny cool"}. '
    "Do not use {'user_id': '...'} directly."
)

item_rag_tool = JSONSearchTool(
    json_path=item_json_path,
    collection_name="v3_hf_item_data",
    config=rag_config
)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = (
    "Search item data. IMPORTANT: input must be "
    '{"search_query": "item_id: <ITEM_ID> name categories stars review_count attributes hours city state"}. '
    "Do not use {'item_id': '...'} directly."
)

review_rag_tool = JSONSearchTool(
    json_path=review_json_path,
    collection_name="v3_hf_review_data",
    config=rag_config
)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = (
    "Search review data. IMPORTANT: input must be "
    '{"search_query": "user_id: <USER_ID> stars text date"} '
    'or {"search_query": "item_id: <ITEM_ID> stars text date"}. '
    "Do not use {'user_id': '...'} or {'item_id': '...'} directly."
)


@CrewBase
class MyProject:
    """MyProject crew"""

    agents_config = agents_config
    tasks_config = tasks_config

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def user_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config["user_profiler"],
            tools=[user_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],
            tools=[item_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],
            tools=[],
            verbose=True,
            allow_delegation=False
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_user_task"],
            agent=self.user_profiler()
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_item_task"],
            agent=self.item_analyst()
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],
            agent=self.prediction_modeler(),
            context=[self.analyze_user_task(), self.analyze_item_task()],
            output_file="report.json"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )