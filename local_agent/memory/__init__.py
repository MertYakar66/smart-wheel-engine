"""Memory and persistence module for the autonomous browser agent"""

from local_agent.memory.chroma_manager import ChromaManager, PageMemory, ExtractedData, TaskPlanMemory
from local_agent.memory.logger import StructuredLogger, StepLog, TaskLog

__all__ = [
    "ChromaManager",
    "PageMemory",
    "ExtractedData",
    "TaskPlanMemory",
    "StructuredLogger",
    "StepLog",
    "TaskLog",
]
