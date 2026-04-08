"""
Pydantic models for SupportDesk-OpenEnv.
Defines typed Observation, Action, and Reward models per the OpenEnv spec.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    What the agent sees at each step.
    Contains the current email to process plus task context.
    """
    email_id: str = Field(..., description="Unique identifier for this email")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    sender: str = Field(..., description="Display name of the sender")
    sender_email: str = Field(..., description="Sender's email address")
    timestamp: str = Field(..., description="ISO-8601 timestamp when email was received")
    thread_length: int = Field(..., description="Number of messages in the thread")
    attachments: List[str] = Field(default_factory=list, description="List of attachment filenames")

    # Task context
    task_name: str = Field(..., description="Active task: email_classify | email_extract | email_respond")
    task_instruction: str = Field(..., description="Human-readable instructions for the agent")
    current_step: int = Field(..., description="Current step within the episode (1-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    episode_id: str = Field(..., description="Unique identifier for this episode")

    # Optional knowledge base (provided for email_respond task)
    knowledge_base: Optional[Dict[str, str]] = Field(
        default=None,
        description="Knowledge base articles relevant to this email (only for email_respond task)"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """
    What the agent submits at each step.
    The relevant fields depend on the active task.
    """
    action_type: str = Field(
        ...,
        description=(
            "Type of action: "
            "'classify' (for email_classify task), "
            "'extract' (for email_extract task), "
            "'respond' (for email_respond task)"
        )
    )

    # --- Fields for 'classify' action ---
    priority: Optional[str] = Field(
        default=None,
        description="Predicted priority: 'urgent' | 'high' | 'normal' | 'low'"
    )
    category: Optional[str] = Field(
        default=None,
        description="Predicted category: 'billing' | 'technical' | 'account' | 'general' | 'complaint'"
    )

    # --- Fields for 'extract' action ---
    extracted_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured info extracted from the email. Expected keys: "
            "customer_id (str), issue_type (str), urgency_signals (List[str]), "
            "affected_product (str), requested_action (str)"
        )
    )

    # --- Fields for 'respond' action ---
    response_text: Optional[str] = Field(
        default=None,
        description="The full text of the composed email response"
    )


# ---------------------------------------------------------------------------
# Reward / Step result
# ---------------------------------------------------------------------------

class EmailReward(BaseModel):
    """
    Returned by step() — contains reward signal and episode metadata.
    """
    reward: float = Field(..., description="Reward for this step [0.0, 1.0]")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic info from the grader")


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """
    Returned by state() — full internal state of the environment.
    """
    episode_id: str
    task_name: str
    current_step: int
    max_steps: int
    total_reward: float
    cumulative_rewards: List[float]
    done: bool
    emails_processed: int
    emails_total: int
    last_action: Optional[Dict[str, Any]] = None
    last_grader_feedback: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# API request/response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = Field(
        default="email_classify",
        description="Task to run: 'email_classify' | 'email_extract' | 'email_respond'"
    )
    session_id: Optional[str] = Field(default=None, description="Optional session ID for tracking")


class StepRequest(BaseModel):
    action: EmailAction
    session_id: str = Field(..., description="Session ID returned by /reset")


class StepResponse(BaseModel):
    observation: Optional[EmailObservation]
    reward: float
    done: bool
    info: Dict[str, Any]
