import pytest
from datetime import datetime, timedelta
from typing import List

from src.social.models import Post, Comment
from src.social.service import SocialFeedService


@pytest.fixture
def service() -> SocialFeedService:
    """Fixture providing a clean SocialFeedService instance for each test."""
    return SocialFeedService()


@pytest.fixture
def sample_post(service: SocialFeedService) -> Post:
    """Fixture providing an existing post in the social feed."""
    return service.create_post(author_id="author_1", content="First social post")


# ============================================================================
# Acceptance Criteria 1: Likes, Reposts, and User Profile Association
# ============================================================================


def test_like_post_increments_count(service: SocialFeedService, sample_post: Post):
    """
    Given an existing post in the social feed,
    When a user likes the post,
    Then the post's like interaction count is incremented.
    """
    assert sample_post.likes_count == 0

    service.like_post(user_id="user_1", post_id=sample_post.id)
    post_after_first_like = service.get_post(sample_post.id)
    assert post_after_first_like.likes_count == 1

    service.like_post(user_id="user_2", post_id=sample_post.id)
    post_after_second_like = service.get_post(sample_post.id)
    assert post_after_second_like.likes_count == 2


def test_like_post_associates_with_user_profile(
    service: SocialFeedService, sample_post: Post
):
    """
    Given an existing post,
    When a user likes the post,
    Then the like interaction is associated with the user's profile.
    """
    user_id = "user_like_tracker"
    service.like_post(user_id=user_id, post_id=sample_post.id)

    user_likes = service.get_user_likes(user_id=user_id)
    assert sample_post.id in user_likes


def test_duplicate_like_by_same_user_raises_error(
    service: SocialFeedService, sample_post: Post
):
    """
    Given a user has already liked a post,
    When the same user attempts to like the post again,
    Then a ValueError is raised and the like count remains unchanged.
    """
    user_id = "user_repeat"
    service.like_post(user_id=user_id, post_id=sample_post.id)

    with pytest.raises(ValueError):
        service.like_post(user_id=user_id, post_id=sample_post.id)

    post = service.get_post(sample_post.id)
    assert post.likes_count == 1


def test_like_nonexistent_post_raises_error(service: SocialFeedService):
    """
    Given a non-existent post ID,
    When a user attempts to like the post,
    Then a ValueError is raised.
    """
    with pytest.raises(ValueError):
        service.like_post(user_id="user_1", post_id="nonexistent_post_id")


def test_repost_post_increments_count(
    service: SocialFeedService, sample_post: Post
):
    """
    Given an existing post in the social feed,
    When a user reposts the post,
    Then the repost interaction count is incremented.
    """
    assert sample_post.reposts_count == 0

    service.repost_post(user_id="user_1", post_id=sample_post.id)
    post_after_first_repost = service.get_post(sample_post.id)
    assert post_after_first_repost.reposts_count == 1

    service.repost_post(user_id="user_2", post_id=sample_post.id)
    post_after_second_repost = service.get_post(sample_post.id)
    assert post_after_second_repost.reposts_count == 2


def test_repost_post_associates_with_user_profile(
    service: SocialFeedService, sample_post: Post
):
    """
    Given an existing post,
    When a user reposts the post,
    Then the repost interaction is associated with the user's profile.
    """
    user_id = "user_repost_tracker"
    service.repost_post(user_id=user_id, post_id=sample_post.id)

    user_reposts = service.get_user_reposts(user_id=user_id)
    assert sample_post.id in user_reposts


def test_duplicate_repost_by_same_user_raises_error(
    service: SocialFeedService, sample_post: Post
):
    """
    Given a user has already reposted a post,
    When the same user attempts to repost it again,
    Then a ValueError is raised and the repost count does not increment.
    """
    user_id = "user_repost_dup"
    service.repost_post(user_id=user_id, post_id=sample_post.id)

    with pytest.raises(ValueError):
        service.repost_post(user_id=user_id, post_id=sample_post.id)

    post = service.get_post(sample_post.id)
    assert post.reposts_count == 1


def test_repost_nonexistent_post_raises_error(service: SocialFeedService):
    """
    Given a non-existent post ID,
    When a user attempts to repost the post,
    Then a ValueError is raised.
    """
    with pytest.raises(ValueError):
        service.repost_post(user_id="user_1", post_id="nonexistent_post_id")


# ============================================================================
# Acceptance Criteria 2: Threaded Comments and Hierarchical Tree
# ============================================================================


def test_add_top_level_comment(service: SocialFeedService, sample_post: Post):
    """
    Given an existing post,
    When a user submits a comment with parent_id=None,
    Then a top-level comment is created for that post.
    """
    comment = service.add_comment(
        post_id=sample_post.id,
        author_id="user_c1",
        content="Top level comment",
        parent_comment_id=None,
    )

    assert comment.id is not None
    assert comment.post_id == sample_post.id
    assert comment.author_id == "user_c1"
    assert comment.content == "Top level comment"
    assert comment.parent_comment_id is None
    assert isinstance(comment.created_at, datetime)


def test_add_reply_to_comment_preserves_parent_id(
    service: SocialFeedService, sample_post: Post
):
    """
    Given an existing comment on a post,
    When a user submits a reply referencing that comment as parent,
    Then the threaded comment correctly sets parent_comment_id.
    """
    root_comment = service.add_comment(
        post_id=sample_post.id,
        author_id="user_c1",
        content="Root comment",
    )

    reply = service.add_comment(
        post_id=sample_post.id,
        author_id="user_c2",
        content="First reply",
        parent_comment_id=root_comment.id,
    )

    assert reply.parent_comment_id == root_comment.id
    assert reply.post_id == sample_post.id


def test_deeply_nested_threaded_comment_hierarchy(
    service: SocialFeedService, sample_post: Post
):
    """
    Given a post with multi-level replies (root -> child -> grandchild),
    When the comments tree is retrieved,
    Then the hierarchical structure is correctly preserved at each depth.
    """
    root = service.add_comment(
        post_id=sample_post.id, author_id="u1", content="L1: Root"
    )
    child = service.add_comment(
        post_id=sample_post.id,
        author_id="u2",
        content="L2: Child",
        parent_comment_id=root.id,
    )
    grandchild = service.add_comment(
        post_id=sample_post.id,
        author_id="u3",
        content="L3: Grandchild",
        parent_comment_id=child.id,
    )

    tree = service.get_comment_tree(post_id=sample_post.id)

    assert len(tree) == 1
    assert tree[0].id == root.id
    assert len(tree[0].replies) == 1
    assert tree[0].replies[0].id == child.id
    assert len(tree[0].replies[0].replies) == 1
    assert tree[0].replies[0].replies[0].id == grandchild.id


def test_add_comment_with_invalid_parent_id_raises_error(
    service: SocialFeedService, sample_post: Post
):
    """
    Given a post,
    When a user submits a reply with a non-existent parent comment ID,
    Then a ValueError is raised.
    """
    with pytest.raises(ValueError):
        service.add_comment(
            post_id=sample_post.id,
            author_id="user_1",
            content="Orphan reply",
            parent_comment_id="nonexistent_parent_id",
        )


def test_add_comment_to_nonexistent_post_raises_error(
    service: SocialFeedService,
):
    """
    Given a non-existent post ID,
    When a user submits a comment,
    Then a ValueError is raised.
    """
    with pytest.raises(ValueError):
        service.add_comment(
            post_id="nonexistent_post",
            author_id="user_1",
            content="Comment to nowhere",
        )


def test_reply_referencing_comment_from_different_post_raises_error(
    service: SocialFeedService,
):
    """
    Given two distinct posts and a comment on Post A,
    When a reply is submitted to Post B targeting Post A's comment as parent,
    Then a ValueError is raised to prevent cross-post corruption.
    """
    post_a = service.create_post(author_id="user_a", content="Post A")
    post_b = service.create_post(author_id="user_b", content="Post B")

    comment_a = service.add_comment(
        post_id=post_a.id, author_id="user_1", content="Comment on A"
    )

    with pytest.raises(ValueError):
        service.add_comment(
            post_id=post_b.id,
            author_id="user_2",
            content="Cross post reply",
            parent_comment_id=comment_a.id,
        )


# ============================================================================
# Acceptance Criteria 3: Social Feed Aggregation and Nested Threads
# ============================================================================


def test_feed_aggregates_metrics_correctly(service: SocialFeedService):
    """
    Given posts with varying likes and reposts,
    When the feed is requested,
    Then each post in the feed returns correct aggregated interaction metrics.
    """
    post1 = service.create_post(author_id="u1", content="Post 1")
    post2 = service.create_post(author_id="u2", content="Post 2")

    # Post 1 gets 2 likes and 1 repost
    service.like_post(user_id="liker_1", post_id=post1.id)
    service.like_post(user_id="liker_2", post_id=post1.id)
    service.repost_post(user_id="reposter_1", post_id=post1.id)

    # Post 2 gets 0 likes and 2 reposts
    service.repost_post(user_id="reposter_2", post_id=post2.id)
    service.repost_post(user_id="reposter_3", post_id=post2.id)

    feed = service.get_feed()
    feed_by_id = {item.id: item for item in feed}

    assert feed_by_id[post1.id].likes_count == 2
    assert feed_by_id[post1.id].reposts_count == 1
    assert feed_by_id[post2.id].likes_count == 0
    assert feed_by_id[post2.id].reposts_count == 2


def test_feed_includes_nested_comment_threads(service: SocialFeedService):
    """
    Given posts with threaded comments,
    When the feed is requested,
    Then each post contains its nested comment threads rather than flat lists.
    """
    post = service.create_post(author_id="u1", content="Feed post with threads")

    c1 = service.add_comment(
        post_id=post.id, author_id="u2", content="Top comment 1"
    )
    c2 = service.add_comment(
        post_id=post.id, author_id="u3", content="Top comment 2"
    )
    r1_c1 = service.add_comment(
        post_id=post.id,
        author_id="u4",
        content="Reply to comment 1",
        parent_comment_id=c1.id,
    )

    feed = service.get_feed()
    feed_post = next(p for p in feed if p.id == post.id)

    # Top-level comments on the post should only be c1 and c2
    assert len(feed_post.comments) == 2
    top_comment_ids = [c.id for c in feed_post.comments]
    assert c1.id in top_comment_ids
    assert c2.id in top_comment_ids
    assert r1_c1.id not in top_comment_ids

    # Nested reply should be inside c1's replies
    c1_in_feed = next(c for c in feed_post.comments if c.id == c1.id)
    assert len(c1_in_feed.replies) == 1
    assert c1_in_feed.replies[0].id == r1_c1.id


def test_feed_comments_ordered_by_creation_time(service: SocialFeedService):
    """
    Given comments and replies created chronologically,
    When the feed is requested,
    Then top-level comments and nested replies are ordered by creation time.
    """
    post = service.create_post(author_id="u1", content="Ordered post")

    # Create root comments with explicit timestamps or sequential creation
    c1 = service.add_comment(
        post_id=post.id, author_id="u2", content="Earlier top comment"
    )
    c2 = service.add_comment(
        post_id=post.id, author_id="u3", content="Later top comment"
    )

    # Create replies under c1 in sequence
    r1 = service.add_comment(
        post_id=post.id,
        author_id="u4",
        content="First reply",
        parent_comment_id=c1.id,
    )
    r2 = service.add_comment(
        post_id=post.id,
        author_id="u5",
        content="Second reply",
        parent_comment_id=c1.id,
    )

    feed = service.get_feed()
    feed_post = next(p for p in feed if p.id == post.id)

    # Validate top-level chronological ordering
    assert feed_post.comments[0].id == c1.id
    assert feed_post.comments[1].id == c2
    assert (
        feed_post.comments[0].created_at <= feed_post.comments[1].created_at
    )

    # Validate child reply chronological ordering
    c1_replies = feed_post.comments[0].replies
    assert len(c1_replies) == 2
    assert c1_replies[0].id == r1.id
    assert c1_replies[1].id == r2.id
    assert c1_replies[0].created_at <= c1_replies[1].created_at


def test_feed_posts_ordered_by_creation_time_descending(
    service: SocialFeedService,
):
    """
    Given multiple posts created at different times,
    When the feed is requested,
    Then posts are returned ordered by creation time (newest first).
    """
    post_old = service.create_post(author_id="u1", content="Older post")
    post_new = service.create_post(author_id="u2", content="Newer post")

    feed = service.get_feed()

    post_indices = {item.id: idx for idx, item in enumerate(feed)}
    assert post_indices[post_new.id] < post_indices[post_old.id]


def test_empty_feed_returns_empty_list(service: SocialFeedService):
    """
    Given no posts in the system,
    When the feed is requested,
    Then an empty list is returned without error.
    """
    feed = service.get_feed()
    assert isinstance(feed, list)
    assert len(feed) == 0


def test_feed_post_with_zero_interactions_and_no_comments(
    service: SocialFeedService, sample_post: Post
):
    """
    Given a newly created post with no interactions or comments,
    When the feed is requested,
    Then interaction counts are 0 and comments list is empty.
    """
    feed = service.get_feed()
    assert len(feed) == 1
    feed_item = feed[0]

    assert feed_item.id == sample_post.id
    assert feed_item.likes_count == 0
    assert feed_item.reposts_count == 0
    assert feed_item.comments == []