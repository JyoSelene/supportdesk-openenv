---
title: SupportDesk-OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# SupportDesk-OpenEnv

**A real-world customer email triage environment for AI agent training and evaluation.**

SupportDesk-OpenEnv puts an agent in the role of a customer support specialist processing an inbox of realistic emails. The agent must classify tickets by urgency and type, extract structured information for CRM entry, and compose accurate responses using a knowledge base — the exact skills a human support agent develops over months on the job.

---

## Why This Environment?

Customer support is a high-volume, economically critical workflow where AI can provide significant leverage. Unlike toy classification tasks, it requires:

- **Contextual understanding** (tone, urgency signals, hidden intent)
- **Structured extraction** from free-form text
- **Grounded generation** (responses must be factually accurate against a KB)
- **Partial progress measurement** (most emails are *somewhat* correct, not binary pass/fail)

SupportDesk-OpenEnv provides a challenging, multi-skill benchmark that maps directly to real agent deployment scenarios.

---

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `email_classify` | Easy | 10 | Classify each email's **priority** (urgent/high/normal/low) and **category** (billing/technical/account/general/complaint) |
| `email_extract` | Medium | 6 | Extract 5 structured fields: customer_id, issue_type, urgency_signals, affected_product, requested_action |
| `email_respond` | Hard | 4 | Compose a professional, factually accurate response using the provided knowledge base |

### Task 1 — email_classify (Easy)

The agent receives one email per step and must predict two labels. Grading provides **partial credit** for priority: getting "high" when the answer is "urgent" earns 0.5 rather than 0.

```json
// Example action
{
  "action_type": "classify",
  "priority": "urgent",
  "category": "billing"
}
```

**Reward formula**: `0.6 × priority_score + 0.4 × category_score`

Priority scoring:
- Exact match → 1.0
- Off by 1 level → 0.5
- Off by 2 levels → 0.2
- Off by 3+ levels → 0.0

### Task 2 — email_extract (Medium)

The agent must extract a structured ticket record from each email. Fields are weighted by importance:

| Field | Weight | Matching |
|-------|--------|---------|
| customer_id | 20% | Exact string match |
| issue_type | 25% | Token Jaccard similarity |
| urgency_signals | 25% | F1 over fuzzy signal matching |
| affected_product | 15% | Token Jaccard similarity |
| requested_action | 15% | Token Jaccard similarity |

```json
// Example action
{
  "action_type": "extract",
  "extracted_info": {
    "customer_id": "CUS-78234",
    "issue_type": "billing_dispute",
    "urgency_signals": ["double-charged", "chargeback threat", "end of day deadline"],
    "affected_product": "premium_subscription",
    "requested_action": "refund"
  }
}
```

### Task 3 — email_respond (Hard)

The agent receives each email alongside knowledge base articles and must write a complete, helpful reply. Grading is deterministic — no LLM judge required:

| Criterion | Weight | How Measured |
|-----------|--------|-------------|
| Required terms present | 40% | Fraction of KB-derived terms found in response |
| No forbidden phrases | 15% | Binary: all absent → 1.0 |
| Professional format | 20% | Greeting present (10%) + Closing present (10%) |
| Factual accuracy | 25% | Fraction of correct facts from KB found in response |

```json
// Example action
{
  "action_type": "respond",
  "response_text": "Dear Derek,\n\nThank you for reaching out! ..."
}
```

---

## Observation Space

Every `step()` call returns an `EmailObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | string | Unique identifier |
| `subject` | string | Email subject |
| `body` | string | Full email body |
| `sender` | string | Sender name |
| `sender_email` | string | Sender address |
| `timestamp` | string | ISO-8601 |
| `thread_length` | int | Replies in thread |
| `attachments` | List[str] | Attachment names |
| `task_name` | string | Active task |
| `task_instruction` | string | Full formatted prompt for the agent |
| `current_step` | int | Current step (1-indexed) |
| `max_steps` | int | Total emails in episode |
| `episode_id` | string | Episode UUID |
| `knowledge_base` | dict\|null | KB articles (respond task only) |

---

## Action Space

| Field | Type | Used by task |
|-------|------|--------------|
| `action_type` | `"classify"\|"extract"\|"respond"` | All |
| `priority` | `"urgent"\|"high"\|"normal"\|"low"` | classify |
| `category` | `"billing"\|"technical"\|"account"\|"general"\|"complaint"` | classify |
| `extracted_info` | object with 5 fields | extract |
| `response_text` | string | respond |

---

## Reward Function

- **Per-step reward**: [0.0, 1.0], returned by task-specific grader
- **Episode score**: mean of all per-step rewards
- **Partial progress**: All three graders provide smooth partial credit signals — not just binary success/failure
- **Trivial action penalty**: −0.05 for empty or clearly placeholder actions

---

## API Reference

The environment exposes a REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome + task list |
| `/health` | GET | Liveness probe |
| `/tasks` | GET | Task metadata |
| `/reset` | POST | Start episode → returns `session_id` + first observation |
| `/step` | POST | Submit action → returns next obs, reward, done, info |
| `/state` | GET | Full environment state |
| `/openenv.yaml` | GET | OpenEnv spec |

### Quick example

```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "email_classify"}'

# Submit action (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<your-session-id>",
    "action": {
      "action_type": "classify",
      "priority": "urgent",
      "category": "billing"
    }
  }'
```

---

## Setup & Usage

### Option 1 — Docker (recommended)

```bash
# Build
docker build -t supportdesk-openenv .

# Run
docker run -p 7860:7860 supportdesk-openenv

# Environment is now available at http://localhost:7860
```

### Option 2 — Local Python

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Running the Baseline Inference Script

```bash
# Required: set environment variables
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SUPPORTDESK_URL=http://localhost:7860  # or your HF Space URL

python inference.py
```

Expected output format:

```
[START] task=email_classify env=supportdesk-openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"classify","priority":"urgent","category":"billing"} reward=1.00 done=false error=null
...
[END] success=true steps=10 score=0.72 rewards=1.00,0.60,0.80,...
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via the HF inference router (temperature=0):

| Task | Score | Notes |
|------|-------|-------|
| email_classify | ~0.72 | Strong on category; occasional priority confusion (urgent vs. high) |
| email_extract | ~0.55 | Good on customer_id; weaker on urgency_signals list |
| email_respond | ~0.48 | Consistent format; sometimes misses specific KB terms |
| **Average** | **~0.58** | |

---

## Project Structure

```
supportdesk-openenv/
├── email_triage_env/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic models (Observation, Action, Reward, State)
│   ├── data.py           # Synthetic email dataset with ground truth labels
│   ├── graders.py        # Deterministic task graders (no LLM required)
│   └── env.py            # SupportDeskEnv class (reset/step/state)
├── app.py                # FastAPI server
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv spec
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Hugging Face Space

The environment is deployed as a Docker Space. To deploy your own fork:

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repository to the Space

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/supportdesk-openenv
git push space main
```

Tags to add in your Space settings: `openenv`, `nlp`, `customer-support`, `rl`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server port |
| `HF_TOKEN` | — | Hugging Face API key (for inference.py) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `SUPPORTDESK_URL` | `http://localhost:7860` | Environment URL for inference.py |

---

## License

MIT
