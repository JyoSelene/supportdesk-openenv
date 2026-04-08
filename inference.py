"""
inference.py — Baseline inference script for SupportDesk-OpenEnv.

Runs an LLM agent (via OpenAI-compatible API) against all three tasks and
emits structured logs in the mandatory [START] / [STEP] / [END] format.

Environment variables
---------------------
HF_TOKEN       : Hugging Face API token (used as API key)
API_BASE_URL   : OpenAI-compatible base URL (default: HF inference router)
MODEL_NAME     : Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
SUPPORTDESK_URL: URL of the running SupportDesk environment
                 (default: http://localhost:7860)

Usage
-----
    # Against local server
    python inference.py

    # Against HF Space
    SUPPORTDESK_URL=https://your-space.hf.space python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL: str = (
    os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
)
MODEL_NAME: str = (
    os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
)
ENV_URL: str = (
    os.getenv("SUPPORTDESK_URL") or "http://localhost:7860"
).rstrip("/")

BENCHMARK = "supportdesk-openenv"
MAX_STEPS = 15          # safety cap per episode
MAX_RETRIES = 2         # retries on parse errors
TEMPERATURE = 0.0       # deterministic for reproducibility
REQUEST_TIMEOUT = 30    # seconds for env HTTP calls

TASKS = ["email_classify", "email_extract", "email_respond"]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_name: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_state(session_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert customer support agent. "
    "Follow the task instructions exactly. "
    "Always respond with a single valid JSON object as specified. "
    "Do not include markdown code fences or explanatory text outside the JSON."
)


def call_llm(task_instruction: str, retry: int = 0) -> str:
    """Call the LLM and return the raw text response."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_instruction},
            ],
            temperature=TEMPERATURE,
            max_tokens=1024,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(1)
            return call_llm(task_instruction, retry + 1)
        return f'{{"error": "{str(e)[:100]}"}}'


def parse_action(raw: str, task_name: str) -> Dict[str, Any]:
    """
    Parse the LLM's raw text into an action dict.
    Falls back gracefully on JSON parse failures.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    try:
        action = json.loads(text)
        # Ensure action_type is set correctly
        if "action_type" not in action:
            if task_name == "email_classify":
                action["action_type"] = "classify"
            elif task_name == "email_extract":
                action["action_type"] = "extract"
            elif task_name == "email_respond":
                action["action_type"] = "respond"
        return action
    except json.JSONDecodeError:
        # Attempt to extract JSON object from the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        # Final fallback: return minimal valid action
        if task_name == "email_classify":
            return {"action_type": "classify", "priority": "normal", "category": "general"}
        elif task_name == "email_extract":
            return {
                "action_type": "extract",
                "extracted_info": {
                    "customer_id": "unknown",
                    "issue_type": "unknown",
                    "urgency_signals": [],
                    "affected_product": "unknown",
                    "requested_action": "unknown",
                },
            }
        else:
            return {
                "action_type": "respond",
                "response_text": "Thank you for contacting support. We will look into your issue.",
            }

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str) -> Dict[str, Any]:
    """
    Run one full episode on a task.

    Returns a summary dict with steps, score, and per-step rewards.
    """
    # Reset environment
    reset_data = env_reset(task_name)
    session_id: str = reset_data["session_id"]
    obs: Dict[str, Any] = reset_data["observation"]

    rewards: List[float] = []
    steps: int = 0
    done: bool = False
    last_error: Optional[str] = None
    final_score: float = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    while not done and steps < MAX_STEPS:
        steps += 1
        task_instruction = obs.get("task_instruction", "")

        # Call LLM
        raw_response = call_llm(task_instruction)
        action = parse_action(raw_response, task_name)

        # Step environment
        try:
            step_data = env_step(session_id, action)
        except requests.HTTPError as e:
            last_error = str(e)[:120]
            print(
                f"[STEP] step={steps} action={json.dumps(action)[:80]} "
                f"reward=0.00 done=true error={last_error}"
            )
            break

        reward: float = step_data.get("reward", 0.0)
        done = step_data.get("done", True)
        info: Dict[str, Any] = step_data.get("info", {})
        next_obs = step_data.get("observation")
        last_error = None

        rewards.append(reward)

        # Capture the final episode score from info when done
        if done and "final_score" in info:
            final_score = info["final_score"]
        elif rewards:
            final_score = round(sum(rewards) / len(rewards), 4)

        # Emit [STEP] line
        action_str = json.dumps(action)[:100].replace("\n", " ")
        error_field = "null" if last_error is None else last_error
        print(
            f"[STEP] step={steps} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_field}"
        )

        if not done and next_obs is not None:
            obs = next_obs

    # Compute final score if not already set
    if not rewards:
        final_score = 0.0
    elif not done:
        # Hit MAX_STEPS — use average so far
        final_score = round(sum(rewards) / len(rewards), 4)

    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )

    return {
        "task": task_name,
        "steps": steps,
        "score": final_score,
        "rewards": rewards,
        "success": success,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"SupportDesk-OpenEnv Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"Env   : {ENV_URL}")
    print(f"Tasks : {TASKS}")
    print("-" * 60)

    # Health check
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"Environment health: {health.json()}")
    except Exception as e:
        print(f"WARNING: Could not reach environment at {ENV_URL}: {e}", file=sys.stderr)
        print("Proceeding anyway — server may be starting up...", file=sys.stderr)

    results = []
    for task in TASKS:
        try:
            result = run_episode(task)
            results.append(result)
        except Exception as e:
            print(f"ERROR running task {task}: {e}", file=sys.stderr)
            # Emit a failed [END] line so the evaluator can still parse
            print(f"[END] success=false steps=0 score=0.00 rewards=")
            results.append({"task": task, "steps": 0, "score": 0.0, "rewards": [], "success": False})

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']:<20} score={r['score']:.4f}  success={r['success']}")
    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"  {'AVERAGE':<20} score={avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
