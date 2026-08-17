"""Very small in-memory conversation store, keyed by session_id.

Good enough for a pre-assessment / single-instance dev server: each session
keeps its last few (question, answer) turns so follow-up questions like
"Who proposed this approach?" can be resolved against prior context.
For a production system this would move to Redis or a DB.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Turn:
    question: str
    answer: str


class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Turn]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[Turn]:
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        history = self._sessions[session_id]
        history.append(Turn(question=question, answer=answer))
        self._sessions[session_id] = history[-self.max_turns :]

    def as_prompt_context(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)
