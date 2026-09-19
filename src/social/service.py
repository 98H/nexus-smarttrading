from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from src.social.models import Comment, Post


class SocialFeedService:
    """Service managing social feed posts, interactions, and threaded comments."""

    def __init__(self) -> None:
        self._posts: Dict[str, Post] = {}
        self._comments: Dict[str, Comment] = {}
        self._post_comments: Dict[str, List[str]] = defaultdict(list)

        self._post_likes: Dict[str, Set[str]] = defaultdict(set)
        self._user_likes: Dict[str, List[str]] = defaultdict(list)

        self._post_reposts: Dict[str, Set[str]] = defaultdict(set)
        self._user_reposts: Dict[str, List[str]] = defaultdict(list)

        self._seq: int = 0
        self._post_seq: Dict[str, int] = {}
        self._comment_seq: Dict[str, int] = {}

    def _normalize_timestamp(self, dt: Optional[datetime] = None) -> datetime:
        """Normalizes datetime to UTC without skewing physical clocks."""
        if dt is None:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def create_post(
        self,
        author_id: str,
        content: str,
        post_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Post:
        """Creates a new post in the social feed."""
        pid = post_id or uuid.uuid4().hex
        timestamp = self._normalize_timestamp(created_at)

        self._seq += 1
        self._post_seq[pid] = self._seq

        post = Post(
            id=pid,
            author_id=author_id,
            content=content,
            created_at=timestamp,
            likes_count=0,
            reposts_count=0,
            comments=[],
        )
        self._posts[pid] = post
        return self.get_post(pid)

    def get_post(self, post_id: str) -> Post:
        """Retrieves an existing post with current interaction metrics and comments."""
        if post_id not in self._posts:
            raise ValueError(f"Post with id '{post_id}' does not exist.")

        post = self._posts[post_id]
        return Post(
            id=post.id,
            author_id=post.author_id,
            content=post.content,
            created_at=post.created_at,
            likes_count=len(self._post_likes[post_id]),
            reposts_count=len(self._post_reposts[post_id]),
            comments=self.get_comment_tree(post_id),
        )

    def like_post(self, user_id: str, post_id: str) -> None:
        """Records a post like for the user and increments interaction metrics."""
        if post_id not in self._posts:
            raise ValueError(f"Post with id '{post_id}' does not exist.")
        if user_id in self._post_likes[post_id]:
            raise ValueError(f"User '{user_id}' has already liked post '{post_id}'.")

        self._post_likes[post_id].add(user_id)
        self._user_likes[user_id].append(post_id)
        self._posts[post_id].likes_count = len(self._post_likes[post_id])

    def get_user_likes(self, user_id: str) -> List[str]:
        """Returns the list of post IDs liked by the specified user."""
        return list(self._user_likes.get(user_id, []))

    def repost_post(self, user_id: str, post_id: str) -> None:
        """Records a repost for the user and increments interaction metrics."""
        if post_id not in self._posts:
            raise ValueError(f"Post with id '{post_id}' does not exist.")
        if user_id in self._post_reposts[post_id]:
            raise ValueError(f"User '{user_id}' has already reposted post '{post_id}'.")

        self._post_reposts[post_id].add(user_id)
        self._user_reposts[user_id].append(post_id)
        self._posts[post_id].reposts_count = len(self._post_reposts[post_id])

    def get_user_reposts(self, user_id: str) -> List[str]:
        """Returns the list of post IDs reposted by the specified user."""
        return list(self._user_reposts.get(user_id, []))

    def add_comment(
        self,
        post_id: str,
        author_id: str,
        content: str,
        parent_comment_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        comment_id: Optional[str] = None,
    ) -> Comment:
        """Creates a comment or reply referencing a parent comment on a post."""
        if post_id not in self._posts:
            raise ValueError(f"Post with id '{post_id}' does not exist.")

        if parent_comment_id is not None:
            if parent_comment_id not in self._comments:
                raise ValueError(
                    f"Parent comment with id '{parent_comment_id}' does not exist."
                )
            parent = self._comments[parent_comment_id]
            if parent.post_id != post_id:
                raise ValueError(
                    f"Parent comment '{parent_comment_id}' does not belong to post '{post_id}'."
                )

        cid = comment_id or uuid.uuid4().hex
        timestamp = self._normalize_timestamp(created_at)

        self._seq += 1
        self._comment_seq[cid] = self._seq

        comment = Comment(
            id=cid,
            post_id=post_id,
            author_id=author_id,
            content=content,
            created_at=timestamp,
            parent_comment_id=parent_comment_id,
            replies=[],
        )
        self._comments[cid] = comment
        self._post_comments[post_id].append(cid)
        self._posts[post_id].comments = self.get_comment_tree(post_id)
        return comment

    def get_comment_tree(self, post_id: str) -> List[Comment]:
        """Builds and returns nested comment threads ordered by creation time."""
        if post_id not in self._posts:
            raise ValueError(f"Post with id '{post_id}' does not exist.")

        comment_ids = self._post_comments.get(post_id, [])
        if not comment_ids:
            return []

        post_comments = [self._comments[cid] for cid in comment_ids]
        post_comments.sort(key=lambda c: (c.created_at, self._comment_seq[c.id]))

        nodes: Dict[str, Comment] = {}
        for c in post_comments:
            nodes[c.id] = Comment(
                id=c.id,
                post_id=c.post_id,
                author_id=c.author_id,
                content=c.content,
                created_at=c.created_at,
                parent_comment_id=c.parent_comment_id,
                replies=[],
            )

        roots: List[Comment] = []
        for c in post_comments:
            node = nodes[c.id]
            if c.parent_comment_id is None:
                roots.append(node)
            else:
                parent_node = nodes.get(c.parent_comment_id)
                if parent_node is not None:
                    parent_node.replies.append(node)

        return roots

    def get_feed(self) -> List[Post]:
        """Returns the feed ordered newest first with aggregated metrics and threads."""
        if not self._posts:
            return []

        sorted_posts = sorted(
            self._posts.values(),
            key=lambda p: (p.created_at, self._post_seq[p.id]),
            reverse=True,
        )

        feed: List[Post] = []
        for p in sorted_posts:
            feed.append(
                Post(
                    id=p.id,
                    author_id=p.author_id,
                    content=p.content,
                    created_at=p.created_at,
                    likes_count=len(self._post_likes[p.id]),
                    reposts_count=len(self._post_reposts[p.id]),
                    comments=self.get_comment_tree(p.id),
                )
            )
        return feed