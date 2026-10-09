"""Quick manual test for the conversational retrieval chain."""

from rag_logic import ConfigError, answer_question


def main() -> None:
    history: list[dict] = []
    print("Chat test started. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            result = answer_question(question, history)
        except ConfigError as exc:
            print("Config error:", exc)
            return
        except Exception as exc:  # surface API errors clearly during testing
            print(f"Error: {type(exc).__name__}: {exc}")
            continue

        print(f"\nAssistant: {result['answer']}")
        for src in result["sources"]:
            print(f"  - {src['source']}, page {src['page']}")
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()