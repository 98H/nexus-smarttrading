from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    content: str
    created_at: datetime
    parent_comment_id: Optional[str] = None
    replies: List[Comment] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Comment):
            return (
                self.id == other.id
                and self.post_id == other.post_id
                and self.author_id == other.author_id
                and self.content == other.content
                and self.created_at == other.created_at
                and self.parent_comment_id == other.parent_comment_id
                and self.replies == other.replies
            )
        if isinstance(other, str):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Post:
    id: str
    author_id: str
    content: str
    created_at: datetime
    likes_count: int = 0
    reposts_count: int = 0
    comments: List[Comment] = field(default_factory=list)