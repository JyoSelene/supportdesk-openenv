"""
SupportDesk-OpenEnv: Customer Email Triage Environment
An OpenEnv-compliant environment for training agents on real-world customer support tasks.
"""
from .env import SupportDeskEnv
from .models import EmailObservation, EmailAction, EmailReward, EnvironmentState

__all__ = ["SupportDeskEnv", "EmailObservation", "EmailAction", "EmailReward", "EnvironmentState"]
__version__ = "1.0.0"
