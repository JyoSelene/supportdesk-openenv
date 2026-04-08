import requests
import json

BASE_URL = "http://localhost:7860"

print("=" * 60)
print("SupportDesk-OpenEnv API Test")
print("=" * 60)

# Test 1: Health
print("\n[1] Testing /health endpoint...")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✓ Status: {resp.status_code}")
    print(f"✓ Response: {resp.json()}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 2: Reset
print("\n[2] Testing /reset endpoint...")
try:
    resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_name": "email_classify"},
        timeout=5
    )
    data = resp.json()
    session_id = data["session_id"]
    email_id = data["observation"]["email_id"]
    subject = data["observation"]["subject"]

    print(f"✓ Session ID: {session_id}")
    print(f"✓ Email ID: {email_id}")
    print(f"✓ Subject: {subject[:50]}...")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 3: Step
print("\n[3] Testing /step endpoint...")
try:
    resp = requests.post(
        f"{BASE_URL}/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "classify",
                "priority": "urgent",
                "category": "billing"
            }
        },
        timeout=5
    )
    result = resp.json()

    print(f"✓ Reward: {result['reward']}")
    print(f"✓ Done: {result['done']}")
    print(f"✓ Score: {result['info']['grader_feedback']['total_score']}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 4: State
print("\n[4] Testing /state endpoint...")
try:
    resp = requests.get(
        f"{BASE_URL}/state",
        params={"session_id": session_id},
        timeout=5
    )
    state = resp.json()

    print(f"✓ Step: {state['current_step']}")
    print(f"✓ Emails processed: {state['emails_processed']}/{state['emails_total']}")
    print(f"✓ Total reward: {state['total_reward']}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
