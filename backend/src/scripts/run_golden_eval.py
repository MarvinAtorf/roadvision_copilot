import json
from pathlib import Path

import requests

GOLDEN_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "stvo_golden_qa.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_eval_results.json"

API_BASE_URL = "http://localhost:8000"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ask_chat_endpoint(question: str) -> str:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"messages": [{"role": "user", "content": question}]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["answer"]


def run_eval(golden_set: list[dict]) -> list[dict]:
    results = []
    for i, item in enumerate(golden_set, 1):
        print(f"[{i}/{len(golden_set)}] {item['paragraph']} ... ", end="", flush=True)
        try:
            actual_answer = ask_chat_endpoint(item["question"])
            status = "ok"
        except requests.exceptions.RequestException as e:
            actual_answer = f"ERROR: {e}"
            status = "failed"
        print(status)

        results.append(
            {
                "paragraph": item["paragraph"],
                "question": item["question"],
                "golden_answer": item["golden_answer"],
                "actual_answer": actual_answer,
                "status": status,
            }
        )
    return results


def main():
    golden_set = load_golden_set()
    results = run_eval(golden_set)

    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{len(results) - len(failed)}/{len(results)} Anfragen erfolgreich.")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Ergebnisse gespeichert: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
