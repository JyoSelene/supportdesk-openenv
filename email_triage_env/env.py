"""
SupportDeskEnv — core environment class.

Implements the OpenEnv interface:
    reset(task_name) → EmailObservation
    step(action)     → (EmailObservation | None, float, bool, dict)
    state()          → EnvironmentState

Episode lifecycle
-----------------
1. reset(task_name) loads the email pool for that task, shuffles it deterministically,
   and returns the first email as an observation.
2. step(action) grades the action for the current email, advances the pointer,
   and returns the next email (or done=True on the last email).
3. state() returns full internal state including cumulative reward.

Reward shaping
--------------
* Per-step reward from grader [0.0, 1.0] (see graders.py for breakdown).
* Episode score = mean of all per-step rewards.
* A small step-efficiency bonus (+0.05, capped at 1.0) is applied when the agent
  solves an email perfectly (reward == 1.0) on the first attempt.
* A repetition penalty (−0.05) is applied if the agent submits an empty or clearly
  trivial action (e.g. no priority in classify).
"""
from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .data import TASK_CONFIG
from .graders import grade
from .models import EmailAction, EmailObservation, EnvironmentState


class SupportDeskEnv:
    """
    Customer-support email triage environment.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible episode ordering.
    """

    VALID_TASKS = ("email_classify", "email_extract", "email_respond")

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        # Episode state (reset on each reset() call)
        self._task_name: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._emails: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._cumulative_rewards: List[float] = []
        self._done: bool = True
        self._last_action: Optional[Dict[str, Any]] = None
        self._last_feedback: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "email_classify") -> EmailObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_name : str
            Task to run: 'email_classify' | 'email_extract' | 'email_respond'.

        Returns
        -------
        EmailObservation — the first email in the episode.
        """
        if task_name not in self.VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {self.VALID_TASKS}"
            )

        config = TASK_CONFIG[task_name]
        emails = copy.deepcopy(config["emails"])
        self._rng.shuffle(emails)

        self._task_name = task_name
        self._episode_id = str(uuid.uuid4())[:8]
        self._emails = emails
        self._current_index = 0
        self._cumulative_rewards = []
        self._done = False
        self._last_action = None
        self._last_feedback = None

        return self._build_observation()

    def step(self, action: EmailAction) -> Tuple[Optional[EmailObservation], float, bool, Dict[str, Any]]:
        """
        Apply action to the current email.

        Parameters
        ----------
        action : EmailAction

        Returns
        -------
        (next_observation, reward, done, info)
            next_observation is None when done=True.
        """
        if self._done or self._task_name is None:
            raise RuntimeError("Call reset() before step().")

        current_email = self._emails[self._current_index]
        action_dict = action.model_dump()

        # Detect clearly trivial actions and apply a small penalty
        trivial = self._is_trivial(action_dict)
        raw_score, feedback = grade(self._task_name, action_dict, current_email)

        if trivial:
            raw_score = max(0.0, raw_score - 0.05)
            feedback["trivial_action_penalty"] = -0.05

        reward = round(raw_score, 4)
        self._cumulative_rewards.append(reward)
        self._last_action = action_dict
        self._last_feedback = feedback

        # Advance to next email
        self._current_index += 1
        done = self._current_index >= len(self._emails)
        self._done = done

        # Build info dict
        info: Dict[str, Any] = {
            "grader_feedback": feedback,
            "email_id": current_email["email_id"],
            "step": len(self._cumulative_rewards),
            "episode_score": round(
                sum(self._cumulative_rewards) / len(self._cumulative_rewards), 4
            ),
        }

        if done:
            info["final_score"] = info["episode_score"]
            next_obs = None
        else:
            next_obs = self._build_observation()

        return next_obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return current environment state snapshot."""
        return EnvironmentState(
            episode_id=self._episode_id or "not_started",
            task_name=self._task_name or "none",
            current_step=len(self._cumulative_rewards),
            max_steps=len(self._emails) if self._emails else 0,
            total_reward=round(sum(self._cumulative_rewards), 4),
            cumulative_rewards=list(self._cumulative_rewards),
            done=self._done,
            emails_processed=self._current_index,
            emails_total=len(self._emails),
            last_action=self._last_action,
            last_grader_feedback=self._last_feedback,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> EmailObservation:
        """Build an EmailObservation from the current email record."""
        config = TASK_CONFIG[self._task_name]  # type: ignore[index]
        email = self._emails[self._current_index]

        # Format the task instruction with email fields
        kb = email.get("knowledge_base")
        kb_str = ""
        if kb:
            kb_str = "\n".join(f"[{k}]\n{v}" for k, v in kb.items())

        instruction = config["instruction_template"].format(
            sender=email["sender"],
            sender_email=email["sender_email"],
            subject=email["subject"],
            body=email["body"],
            knowledge_base=kb_str,
        )

        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            sender_email=email["sender_email"],
            timestamp=email["timestamp"],
            thread_length=email["thread_length"],
            attachments=email.get("attachments", []),
            task_name=self._task_name,  # type: ignore[arg-type]
            task_instruction=instruction,
            current_step=len(self._cumulative_rewards) + 1,
            max_steps=len(self._emails),
            episode_id=self._episode_id,  # type: ignore[arg-type]
            knowledge_base=kb,
        )

    def _is_trivial(self, action_dict: Dict[str, Any]) -> bool:
        """Detect clearly trivial / empty actions."""
        action_type = action_dict.get("action_type", "")
        if action_type == "classify":
            return not action_dict.get("priority") and not action_dict.get("category")
        elif action_type == "extract":
            info = action_dict.get("extracted_info") or {}
            return not info
        elif action_type == "respond":
            text = action_dict.get("response_text") or ""
            return len(text.strip()) < 10
        return False
