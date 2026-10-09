"""Run the app with a simulated exhausted LLM quota (for UI testing/demo)."""

import runpy

import rag_logic


def always_out_of_quota(model_name: str):
    raise RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")


rag_logic.get_llm = always_out_of_quota
runpy.run_path("app.py", run_name="__main__")