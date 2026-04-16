#!/usr/bin/env python
import sys
import warnings
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from my_project.crew import MyProject

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def load_first_test_case():
    """
    Load the first test case from data/test_review_subset.json.

    Supports:
    1. JSON array format: [{...}, {...}]
    2. JSONL format: one JSON object per line
    """
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "test_review_subset.json"
    )

    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty file: {data_path}")

    if content.startswith("["):
        data = json.loads(content)
        if not data:
            raise ValueError(f"No test cases found in: {data_path}")
        return data[0]

    first_line = content.splitlines()[0].strip()
    return json.loads(first_line)


def normalize_result(result):
    """
    Safely convert CrewOutput to a plain dict for printing.
    """
    if hasattr(result, "pydantic") and result.pydantic is not None:
        return result.pydantic.model_dump()

    if hasattr(result, "json_dict") and result.json_dict is not None:
        return result.json_dict

    if hasattr(result, "raw") and result.raw:
        try:
            return json.loads(result.raw)
        except Exception:
            return {"raw": result.raw}

    return {"raw": str(result)}


def run(user_id=None, item_id=None):
    """
    Run the crew on provided IDs or the first test case.
    """
    if user_id is None or item_id is None:
        test_case = load_first_test_case()
        user_id = test_case["user_id"]
        item_id = test_case["item_id"]

    inputs = {
        "user_id": user_id,
        "item_id": item_id,
    }

    print("Running Crew with inputs:")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))

    result = MyProject().crew().kickoff(inputs=inputs)
    final_output = normalize_result(result)

    print("\n=== FINAL RESULT ===")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


def train():
    inputs = {
        "current_year": str(datetime.now().year)
    }

    try:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python main.py train <n_iterations> <filename>")

        MyProject().crew().train(
            n_iterations=int(sys.argv[2]),
            filename=sys.argv[3],
            inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python main.py replay <task_id>")

        MyProject().crew().replay(task_id=sys.argv[2])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    inputs = {
        "current_year": str(datetime.now().year)
    }

    try:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python main.py test <n_iterations> <eval_llm>")

        MyProject().crew().test(
            n_iterations=int(sys.argv[2]),
            eval_llm=sys.argv[3],
            inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    Usage:
        python main.py trigger '{"user_id":"abc","item_id":"xyz"}'
    """
    if len(sys.argv) < 3:
        raise ValueError(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON payload provided as argument: {e}")

    if "user_id" not in trigger_payload or "item_id" not in trigger_payload:
        raise ValueError("Trigger payload must contain both 'user_id' and 'item_id'.")

    run(
        user_id=trigger_payload["user_id"],
        item_id=trigger_payload["item_id"]
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run()
    else:
        command = sys.argv[1].lower()

        if command == "run":
            if len(sys.argv) == 4:
                run(user_id=sys.argv[2], item_id=sys.argv[3])
            else:
                run()

        elif command == "trigger":
            run_with_trigger()

        elif command == "train":
            train()

        elif command == "replay":
            replay()

        elif command == "test":
            test()

        else:
            raise ValueError(
                "Unknown command. Use one of: run, trigger, train, replay, test"
            )