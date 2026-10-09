"""Simulate an exhausted LLM quota and verify graceful degradation."""

import rag_logic


def always_out_of_quota(model_name: str):
    raise RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")


rag_logic.get_llm = always_out_of_quota

result = rag_logic.answer_question("Who was the first Governor General of Pakistan?")
print("degraded:", result["degraded"])
print("sources:", len(result["sources"]))
print(result["answer"])