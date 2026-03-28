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

    # JSON array
    if content.startswith("["):
        data = json.loads(content)
        if not data:
            raise ValueError(f"No test cases found in: {data_path}")
        return data[0]

    # JSONL
    first_line = content.splitlines()[0].strip()
    return json.loads(first_line)


def run():
    """
    Run the crew on the first test case.
    """
    test_case = load_first_test_case()

    inputs = {
        "user_id": test_case["user_id"],
        "item_id": test_case["item_id"],
    }

    print("Running Crew with inputs:")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))

    result = MyProject().crew().kickoff(inputs=inputs)

    print("\n=== FINAL RESULT ===")
    print(result)


def train():
    """
    Train the crew for a given number of iterations.
    Usage:
        python main.py train <n_iterations> <filename>
    """
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
    """
    Replay the crew execution from a specific task.
    Usage:
        python main.py replay <task_id>
    """
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python main.py replay <task_id>")

        MyProject().crew().replay(task_id=sys.argv[2])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results.
    Usage:
        python main.py test <n_iterations> <eval_llm>
    """
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
        python main.py trigger '{"key":"value"}'
    """
    if len(sys.argv) < 3:
        raise ValueError("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON payload provided as argument: {e}")

    inputs = {
        "crewai_trigger_payload": trigger_payload
    }

    try:
        result = MyProject().crew().kickoff(inputs=inputs)
        print(result)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run()
    else:
        command = sys.argv[1].lower()

        if command == "run":
            run()
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        elif command == "trigger":
            run_with_trigger()
        else:
            raise ValueError(
                "Unknown command. Use one of: run, train, replay, test, trigger"
            )