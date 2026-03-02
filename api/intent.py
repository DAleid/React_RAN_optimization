"""
Intent Processing API — Vercel Python Serverless Function
POST /api/intent  →  Runs CrewAI agents to parse and plan 5G network intent
"""

import os, sys, json, re
from pathlib import Path
from flask import Flask, request, jsonify

# ── ensure project root is importable ───────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)


def _get_llm():
    from crewai import LLM
    provider = os.getenv("LLM_PROVIDER", "groq")
    groq_key  = os.getenv("GROQ_API_KEY", "")
    oai_key   = os.getenv("OPENAI_API_KEY", "")

    if provider == "openai" and oai_key:
        return LLM(model="openai/gpt-4o-mini", api_key=oai_key, temperature=0.7)
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_key, temperature=0.7)


def _parse_raw(result: object) -> dict:
    raw = re.sub(r"```(?:json)?", "", str(result)).strip().rstrip("`").strip()
    return json.loads(raw)


def _run_intent_pipeline(user_intent: str) -> dict:
    """Full 4-step intent → plan pipeline."""
    from crewai import Crew, Task, Process

    # ── lazy-import agent factories (same as original project) ──────────────
    agent_dir = ROOT / "agents"
    sys.path.insert(0, str(agent_dir))
    from agents.intent_agent    import create_intent_agent
    from agents.planner_agent   import create_planner_agent
    from agents.monitor_agent   import create_monitor_agent
    from agents.optimizer_agent import create_optimizer_agent

    llm = _get_llm()
    intent_agent    = create_intent_agent(llm)
    planner_agent   = create_planner_agent(llm)
    monitor_agent   = create_monitor_agent(llm)
    optimizer_agent = create_optimizer_agent(llm)

    tasks = [
        Task(
            description=(
                f'Parse this 5G network intent: "{user_intent}"\n'
                "Use parse_intent tool. Return ONLY the raw JSON result."
            ),
            expected_output="JSON intent dict.",
            agent=intent_agent,
        ),
        Task(
            description=(
                "Based on the intent analysis, use generate_config tool to create "
                "a 3GPP-compliant 5G network configuration. Return ONLY raw JSON."
            ),
            expected_output="JSON config dict.",
            agent=planner_agent,
        ),
        Task(
            description=(
                "Use get_metrics and check_status tools to analyse current network "
                "health. Return ONLY raw JSON."
            ),
            expected_output="JSON status dict.",
            agent=monitor_agent,
        ),
        Task(
            description=(
                "Based on the monitoring analysis, use execute_action tool to apply "
                "the best optimisation. Return ONLY raw JSON."
            ),
            expected_output="JSON optimization result.",
            agent=optimizer_agent,
        ),
    ]

    crew = Crew(
        agents=[intent_agent, planner_agent, monitor_agent, optimizer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    import time
    delays = [5, 10, 20]
    for attempt in range(3):
        try:
            result = crew.kickoff()
            break
        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ["rate_limit", "ratelimit", "429"])
            if is_rate and attempt < 2:
                time.sleep(delays[attempt])
                continue
            raise

    # Try to parse the final crew output as JSON
    try:
        return _parse_raw(result)
    except Exception:
        return {"raw_output": str(result)}


def _fallback_pipeline(user_intent: str) -> dict:
    """Keyword-only fallback when CrewAI is unavailable."""
    sys.path.insert(0, str(ROOT))
    from tools.intent_tools import _parse_intent_impl, _fallback_intent
    from tools.config_tools import _generate_config_impl

    try:
        intent = _parse_intent_impl(user_intent)
    except Exception:
        intent = _fallback_intent(user_intent, "LLM unavailable", use_keyword_fallback=True)

    try:
        config = _generate_config_impl(intent)
    except Exception:
        config = {"error": "Config generation failed"}

    return {
        "intent":  intent,
        "config":  config,
        "monitor": None,
        "optimization": None,
        "fallback": True,
    }


@app.route("/api/intent", methods=["POST", "OPTIONS"])
def process_intent():
    # CORS pre-flight
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    body = request.get_json(silent=True) or {}
    user_intent = (body.get("intent") or "").strip()

    if not user_intent:
        return jsonify({"error": "intent field is required"}), 400

    try:
        result = _run_intent_pipeline(user_intent)
        resp   = jsonify({"success": True, "result": result})
    except Exception as e:
        # Graceful degradation — still return useful data
        result = _fallback_pipeline(user_intent)
        resp   = jsonify({"success": True, "result": result,
                          "warning": f"Full pipeline failed, used fallback: {str(e)}"})

    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ── local development server (python api/intent.py) ─────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    app.run(host="0.0.0.0", port=8000, debug=True)
