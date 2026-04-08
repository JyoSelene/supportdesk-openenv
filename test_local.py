#!/usr/bin/env python
"""
test_local.py — Quick local test of all environment components.

Run this to verify the environment works before submission:
    python test_local.py
"""
import sys
import json
from email_triage_env import SupportDeskEnv
from email_triage_env.models import EmailAction
from email_triage_env.data import CLASSIFY_EMAILS, EXTRACT_EMAILS, RESPOND_EMAILS

def test_env():
    """Test all three tasks with perfect actions."""
    print("=" * 70)
    print("SupportDesk-OpenEnv — Local Test Suite")
    print("=" * 70)

    tasks_passed = 0
    tasks_total = 3

    # Test 1: email_classify
    print("\n[1/3] Testing email_classify...")
    try:
        env = SupportDeskEnv(seed=42)
        obs = env.reset("email_classify")
        print(f"  ✓ Reset successful. First email: {obs.email_id}")

        # Take one perfect step
        current_email = next(e for e in CLASSIFY_EMAILS if e["email_id"] == obs.email_id)
        action = EmailAction(
            action_type="classify",
            priority=current_email["labels"]["priority"],
            category=current_email["labels"]["category"],
        )
        next_obs, reward, done, info = env.step(action)
        print(f"  ✓ Step successful. Reward: {reward}")
        assert reward == 1.0, f"Expected reward=1.0 for perfect action, got {reward}"
        print(f"  ✓ Perfect reward confirmed (1.0)")

        # Run full episode with correct answers
        rewards = []
        obs = env.reset("email_classify")
        done = False
        while not done:
            current_email = next(e for e in CLASSIFY_EMAILS if e["email_id"] == obs.email_id)
            action = EmailAction(
                action_type="classify",
                priority=current_email["labels"]["priority"],
                category=current_email["labels"]["category"],
            )
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if not done:
                obs = next_obs

        score = sum(rewards) / len(rewards)
        print(f"  ✓ Full episode complete. {len(rewards)} emails, score={score:.4f}")
        assert score == 1.0, f"Expected perfect score, got {score}"
        print(f"  ✓ email_classify PASSED")
        tasks_passed += 1
    except Exception as e:
        print(f"  ✗ email_classify FAILED: {e}")

    # Test 2: email_extract
    print("\n[2/3] Testing email_extract...")
    try:
        env = SupportDeskEnv(seed=42)
        obs = env.reset("email_extract")
        print(f"  ✓ Reset successful. First email: {obs.email_id}")

        # Run full episode with correct extraction
        rewards = []
        obs = env.reset("email_extract")
        done = False
        while not done:
            current_email = next(e for e in EXTRACT_EMAILS if e["email_id"] == obs.email_id)
            action = EmailAction(
                action_type="extract",
                extracted_info=current_email["ground_truth"],
            )
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if not done:
                obs = next_obs

        score = sum(rewards) / len(rewards)
        print(f"  ✓ Full episode complete. {len(rewards)} emails, score={score:.4f}")
        assert score == 1.0, f"Expected perfect score, got {score}"
        print(f"  ✓ email_extract PASSED")
        tasks_passed += 1
    except Exception as e:
        print(f"  ✗ email_extract FAILED: {e}")

    # Test 3: email_respond
    print("\n[3/3] Testing email_respond...")
    try:
        env = SupportDeskEnv(seed=42)
        obs = env.reset("email_respond")
        print(f"  ✓ Reset successful. First email: {obs.email_id}")

        # Run full episode with responses that include all required terms
        rewards = []
        obs = env.reset("email_respond")
        done = False
        while not done:
            current_email = next(e for e in RESPOND_EMAILS if e["email_id"] == obs.email_id)

            # Build a response that includes all required terms and facts
            req_terms = " ".join(current_email["required_terms"])
            facts = " ".join(
                str(v) for v in current_email["correct_facts"].values()
                if not isinstance(v, bool)
            )
            response = (
                f"Dear Customer,\n\n"
                f"Thank you for contacting us. "
                f"Here is what you need to know: {req_terms}. {facts}. "
                f"We are committed to resolving this for you.\n\n"
                f"Best regards,\nCustomer Support Team"
            )

            action = EmailAction(action_type="respond", response_text=response)
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if not done:
                obs = next_obs

        score = sum(rewards) / len(rewards)
        print(f"  ✓ Full episode complete. {len(rewards)} emails, score={score:.4f}")
        # email_respond is harder, so perfect score may be slightly less than 1.0
        # but should be very high (≥0.95)
        assert score >= 0.95, f"Expected high score, got {score}"
        print(f"  ✓ email_respond PASSED")
        tasks_passed += 1
    except Exception as e:
        print(f"  ✗ email_respond FAILED: {e}")

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {tasks_passed}/{tasks_total} tasks passed")
    print("=" * 70)

    if tasks_passed == tasks_total:
        print("✅ All tests passed! Environment is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(test_env())
