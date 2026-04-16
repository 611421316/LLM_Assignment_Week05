import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import JSONSearchTool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field, field_validator
import json
from typing import Any
import re

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


def validate_prediction(output: Any):
    raw = getattr(output, "raw", output)

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    # strip markdown fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # extract first JSON object if extra text exists
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    try:
        data = json.loads(text)

        # normalize common wrong key
        if "predicted_stars" in data and "stars" not in data:
            data["stars"] = data.pop("predicted_stars")

        # ensure exact keys only
        allowed = {"stars", "review"}
        data = {k: v for k, v in data.items() if k in allowed}

        PredictionOutput.model_validate(data)
        return True, data

    except Exception as e:
        return False, (
            "Invalid final output. Return ONLY raw JSON with exactly "
            '{"stars": number, "review": string}. '
            f"Validation error: {e}"
        )

class PredictionOutput(BaseModel):
    stars: float = Field(..., ge=0.5, le=5.0)
    review: str = Field(..., min_length=1)

    @field_validator("stars")
    @classmethod
    def validate_half_step(cls, value: float) -> float:
        doubled = value * 2
        if abs(doubled - round(doubled)) > 1e-9:
            raise ValueError("stars must use 0.5 increments")
        return value



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

    def project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["project_manager"],
            verbose=True,
            allow_delegation=True
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_user_task"],
            agent=self.user_profiler(),
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
            context=[self.analyze_user_task(), self.analyze_item_task()],
            output_pydantic=PredictionOutput,
            guardrail=validate_prediction,
            guardrail_max_retries=3,
            output_file="report.json"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager_agent=self.project_manager(),
            process=Process.hierarchical,
            verbose=True
        )