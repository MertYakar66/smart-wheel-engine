"""
Verification Engine

Uses Claude Code's WebSearch capability to verify candidate headlines
from the financial_news system. Assigns verification_confidence scores
and pushes verified stories to the server.

Design:
- Local system collects candidates via existing connectors/scheduler
- Claude Code runs verification during AM/PM scheduled sessions
- Verified stories (score >= 7) get pushed to server

This module provides:
- Candidate retrieval from database
- Verification result storage
- Push mechanism for verified stories
- Structured output formatting
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


class VerificationStatus(Enum):
    """Status of verification for a candidate."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    PUSHED = "pushed"


@dataclass
class VerificationCandidate:
    """A headline candidate awaiting verification."""
    candidate_id: str
    headline: str
    source: str
    source_type: str  # official, public, premium
    url: str
    published_at: datetime
    tickers: list[str]
    categories: list[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_verification_query(self) -> str:
        """Format as a verification query for Claude Code."""
        tickers_str = ", ".join(self.tickers) if self.tickers else "unknown"
        categories_str = ", ".join(self.categories) if self.categories else "general"

        return f"""Verify this financial headline using web search:

HEADLINE: "{self.headline}"
SOURCE: {self.source} ({self.source_type})
TICKERS: {tickers_str}
CATEGORIES: {categories_str}
PUBLISHED: {self.published_at.isoformat() if self.published_at else 'unknown'}

VERIFICATION TASKS:
1. Search for corroborating sources (official sources preferred)
2. Check if the event/claim is substantiated
3. Identify any conflicting reports
4. Note the quality of sources found

RETURN FORMAT:
- verification_confidence: 0-10 score
- corroborating_sources: list of URLs/sources found
- what_happened: 1-2 sentence factual summary
- why_it_matters: market significance
- affected_assets: list of tickers/assets impacted
- conflicts_found: any contradictory information
- recommendation: PUSH (>=7), HOLD (4-6), REJECT (<4)"""


@dataclass
class VerificationResult:
    """Result of verifying a candidate headline."""
    candidate_id: str
    verification_confidence: int  # 0-10
    status: VerificationStatus

    # Structured note
    what_happened: str
    why_it_matters: str
    affected_assets: list[str]

    # Verification details
    corroborating_sources: list[dict]  # [{name, url, type}]
    conflicts_found: list[str]

    # Metadata
    verified_at: datetime = field(default_factory=datetime.utcnow)
    verified_by: str = "claude_code"

    def should_push(self) -> bool:
        """Determine if this result should be pushed to server."""
        return self.verification_confidence >= 7


@dataclass
class PushPayload:
    """Payload to push to server for verified stories."""
    story_id: str
    title: str
    what_happened: str
    why_it_matters: str
    affected_assets: list[str]
    verification_confidence: int
    sources: list[dict]
    categories: list[str]
    original_headline: str
    original_source: str
    original_url: str
    verified_at: str
    pushed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        """Convert to JSON for API push."""
        return json.dumps(asdict(self), indent=2)


class VerificationEngine:
    """
    Engine for verifying candidate headlines.

    Workflow:
    1. get_pending_candidates() - retrieve unverified headlines
    2. [Claude Code runs WebSearch verification]
    3. store_result() - save verification result
    4. push_verified() - send to server if threshold met
    """

    def __init__(self, db_path: str = "financial_news/data/news.db"):
        """Initialize verification engine."""
        self.db_path = Path(db_path)
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        """Ensure verification tables exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS verification_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    headline TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    url TEXT,
                    published_at TEXT,
                    tickers TEXT,  -- JSON array
                    categories TEXT,  -- JSON array
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS verification_results (
                    result_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    verification_confidence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    what_happened TEXT,
                    why_it_matters TEXT,
                    affected_assets TEXT,  -- JSON array
                    corroborating_sources TEXT,  -- JSON array
                    conflicts_found TEXT,  -- JSON array
                    verified_at TEXT NOT NULL,
                    verified_by TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES verification_candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS push_log (
                    push_id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    pushed_at TEXT NOT NULL,
                    push_status TEXT,
                    response TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_status
                ON verification_candidates(status);

                CREATE INDEX IF NOT EXISTS idx_results_confidence
                ON verification_results(verification_confidence);
            """)
            conn.commit()
        finally:
            conn.close()

    def add_candidate(
        self,
        headline: str,
        source: str,
        source_type: str,
        url: str = "",
        published_at: datetime | None = None,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> str:
        """
        Add a candidate headline for verification.

        Returns candidate_id.
        """
        # Generate deterministic ID from headline + source
        hash_input = f"{headline}|{source}|{url}"
        candidate_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO verification_candidates
                (candidate_id, headline, source, source_type, url,
                 published_at, tickers, categories, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate_id,
                headline,
                source,
                source_type,
                url,
                published_at.isoformat() if published_at else None,
                json.dumps(tickers or []),
                json.dumps(categories or []),
                datetime.utcnow().isoformat(),
                "pending",
            ))
            conn.commit()
        finally:
            conn.close()

        return candidate_id

    def get_pending_candidates(
        self,
        limit: int = 10,
        max_age_hours: int = 24,
    ) -> list[VerificationCandidate]:
        """
        Get candidates pending verification.

        Args:
            limit: Maximum candidates to return
            max_age_hours: Only return candidates newer than this

        Returns:
            List of VerificationCandidate objects
        """
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()

        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM verification_candidates
                WHERE status = 'pending'
                AND created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()

            candidates = []
            for row in rows:
                candidates.append(VerificationCandidate(
                    candidate_id=row["candidate_id"],
                    headline=row["headline"],
                    source=row["source"],
                    source_type=row["source_type"],
                    url=row["url"] or "",
                    published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                    tickers=json.loads(row["tickers"]) if row["tickers"] else [],
                    categories=json.loads(row["categories"]) if row["categories"] else [],
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))

            return candidates
        finally:
            conn.close()

    def store_result(self, result: VerificationResult) -> None:
        """Store verification result and update candidate status."""
        result_id = f"vr_{result.candidate_id}_{int(datetime.utcnow().timestamp())}"

        conn = self._get_conn()
        try:
            # Store result
            conn.execute("""
                INSERT INTO verification_results
                (result_id, candidate_id, verification_confidence, status,
                 what_happened, why_it_matters, affected_assets,
                 corroborating_sources, conflicts_found, verified_at, verified_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                result.candidate_id,
                result.verification_confidence,
                result.status.value,
                result.what_happened,
                result.why_it_matters,
                json.dumps(result.affected_assets),
                json.dumps(result.corroborating_sources),
                json.dumps(result.conflicts_found),
                result.verified_at.isoformat(),
                result.verified_by,
            ))

            # Update candidate status
            conn.execute("""
                UPDATE verification_candidates
                SET status = ?
                WHERE candidate_id = ?
            """, (result.status.value, result.candidate_id))

            conn.commit()
        finally:
            conn.close()

    def get_pushable_stories(self) -> list[tuple[VerificationCandidate, VerificationResult]]:
        """Get verified stories ready to push (confidence >= 7, not yet pushed)."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT c.*, r.*
                FROM verification_candidates c
                JOIN verification_results r ON c.candidate_id = r.candidate_id
                WHERE r.verification_confidence >= 7
                AND c.status = 'verified'
                ORDER BY r.verified_at DESC
            """).fetchall()

            results = []
            for row in rows:
                candidate = VerificationCandidate(
                    candidate_id=row["candidate_id"],
                    headline=row["headline"],
                    source=row["source"],
                    source_type=row["source_type"],
                    url=row["url"] or "",
                    published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                    tickers=json.loads(row["tickers"]) if row["tickers"] else [],
                    categories=json.loads(row["categories"]) if row["categories"] else [],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )

                result = VerificationResult(
                    candidate_id=row["candidate_id"],
                    verification_confidence=row["verification_confidence"],
                    status=VerificationStatus(row["status"]),
                    what_happened=row["what_happened"],
                    why_it_matters=row["why_it_matters"],
                    affected_assets=json.loads(row["affected_assets"]) if row["affected_assets"] else [],
                    corroborating_sources=json.loads(row["corroborating_sources"]) if row["corroborating_sources"] else [],
                    conflicts_found=json.loads(row["conflicts_found"]) if row["conflicts_found"] else [],
                    verified_at=datetime.fromisoformat(row["verified_at"]),
                    verified_by=row["verified_by"],
                )

                results.append((candidate, result))

            return results
        finally:
            conn.close()

    def create_push_payload(
        self,
        candidate: VerificationCandidate,
        result: VerificationResult,
    ) -> PushPayload:
        """Create payload for pushing to server."""
        story_id = f"story_{candidate.candidate_id}_{int(result.verified_at.timestamp())}"

        return PushPayload(
            story_id=story_id,
            title=result.what_happened[:100] if result.what_happened else candidate.headline[:100],
            what_happened=result.what_happened,
            why_it_matters=result.why_it_matters,
            affected_assets=result.affected_assets,
            verification_confidence=result.verification_confidence,
            sources=result.corroborating_sources,
            categories=candidate.categories,
            original_headline=candidate.headline,
            original_source=candidate.source,
            original_url=candidate.url,
            verified_at=result.verified_at.isoformat(),
        )

    def log_push(
        self,
        payload: PushPayload,
        status: str,
        response: str = "",
    ) -> None:
        """Log a push attempt."""
        push_id = f"push_{payload.story_id}_{int(datetime.utcnow().timestamp())}"

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO push_log
                (push_id, story_id, payload, pushed_at, push_status, response)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                push_id,
                payload.story_id,
                payload.to_json(),
                datetime.utcnow().isoformat(),
                status,
                response,
            ))

            # Mark candidate as pushed
            conn.execute("""
                UPDATE verification_candidates
                SET status = 'pushed'
                WHERE candidate_id = ?
            """, (payload.story_id.split("_")[1],))  # Extract candidate_id from story_id

            conn.commit()
        finally:
            conn.close()

    def get_verification_stats(self) -> dict:
        """Get verification statistics."""
        conn = self._get_conn()
        try:
            stats = {}

            # Candidate counts by status
            rows = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM verification_candidates
                GROUP BY status
            """).fetchall()
            stats["candidates_by_status"] = {row["status"]: row["count"] for row in rows}

            # Average confidence
            row = conn.execute("""
                SELECT AVG(verification_confidence) as avg_confidence,
                       COUNT(*) as total_verified
                FROM verification_results
            """).fetchone()
            stats["avg_confidence"] = round(row["avg_confidence"] or 0, 1)
            stats["total_verified"] = row["total_verified"]

            # Push stats
            rows = conn.execute("""
                SELECT push_status, COUNT(*) as count
                FROM push_log
                GROUP BY push_status
            """).fetchall()
            stats["pushes_by_status"] = {row["push_status"]: row["count"] for row in rows}

            # Recent high-confidence stories
            rows = conn.execute("""
                SELECT r.what_happened, r.verification_confidence, r.verified_at
                FROM verification_results r
                WHERE r.verification_confidence >= 7
                ORDER BY r.verified_at DESC
                LIMIT 5
            """).fetchall()
            stats["recent_verified"] = [
                {
                    "summary": row["what_happened"][:80] + "..." if len(row["what_happened"] or "") > 80 else row["what_happened"],
                    "confidence": row["verification_confidence"],
                    "verified_at": row["verified_at"],
                }
                for row in rows
            ]

            return stats
        finally:
            conn.close()


# =============================================================================
# VERIFICATION WORKFLOW HELPERS
# =============================================================================

def run_verification_cycle(engine: VerificationEngine, limit: int = 5) -> list[dict]:
    """
    Get pending candidates and format verification queries.

    This returns queries that Claude Code should execute via WebSearch.
    After execution, call process_verification_response() with results.
    """
    candidates = engine.get_pending_candidates(limit=limit)

    queries = []
    for candidate in candidates:
        queries.append({
            "candidate_id": candidate.candidate_id,
            "headline": candidate.headline,
            "source": candidate.source,
            "tickers": candidate.tickers,
            "categories": candidate.categories,
            "verification_query": candidate.to_verification_query(),
        })

    return queries


def process_verification_response(
    engine: VerificationEngine,
    candidate_id: str,
    confidence: int,
    what_happened: str,
    why_it_matters: str,
    affected_assets: list[str],
    sources: list[dict],
    conflicts: list[str] | None = None,
) -> VerificationResult:
    """
    Process Claude Code's verification response and store result.

    Args:
        engine: VerificationEngine instance
        candidate_id: ID of the candidate verified
        confidence: 0-10 verification confidence
        what_happened: Factual summary
        why_it_matters: Market significance
        affected_assets: List of affected tickers
        sources: List of corroborating sources [{name, url, type}]
        conflicts: List of conflicting information found

    Returns:
        VerificationResult
    """
    # Determine status
    if confidence >= 7:
        status = VerificationStatus.VERIFIED
    elif confidence >= 4:
        status = VerificationStatus.NEEDS_REVIEW
    else:
        status = VerificationStatus.REJECTED

    result = VerificationResult(
        candidate_id=candidate_id,
        verification_confidence=confidence,
        status=status,
        what_happened=what_happened,
        why_it_matters=why_it_matters,
        affected_assets=affected_assets,
        corroborating_sources=sources,
        conflicts_found=conflicts or [],
    )

    engine.store_result(result)

    return result


def push_verified_stories(
    engine: VerificationEngine,
    server_url: str = "http://localhost:3000/api/news",
    dry_run: bool = True,
) -> list[dict]:
    """
    Push all verified stories to server.

    Args:
        engine: VerificationEngine instance
        server_url: API endpoint to push to
        dry_run: If True, just return payloads without actually pushing

    Returns:
        List of push results
    """
    pushable = engine.get_pushable_stories()
    results = []

    for candidate, verification in pushable:
        payload = engine.create_push_payload(candidate, verification)

        if dry_run:
            results.append({
                "action": "dry_run",
                "story_id": payload.story_id,
                "payload": asdict(payload),
            })
        else:
            # Actual HTTP push would go here
            # For now, just log it
            try:
                # import requests
                # response = requests.post(server_url, json=asdict(payload))
                # status = "success" if response.ok else "failed"
                status = "simulated_success"
                engine.log_push(payload, status)
                results.append({
                    "action": "pushed",
                    "story_id": payload.story_id,
                    "status": status,
                })
            except Exception as e:
                engine.log_push(payload, "error", str(e))
                results.append({
                    "action": "error",
                    "story_id": payload.story_id,
                    "error": str(e),
                })

    return results


# =============================================================================
# CLI INTERFACE FOR CLAUDE CODE
# =============================================================================

def print_verification_queries(limit: int = 5) -> None:
    """Print verification queries for Claude Code to execute."""
    engine = VerificationEngine()
    queries = run_verification_cycle(engine, limit=limit)

    if not queries:
        print("No pending candidates for verification.")
        return

    print(f"=== {len(queries)} CANDIDATES PENDING VERIFICATION ===\n")

    for i, q in enumerate(queries, 1):
        print(f"--- Candidate {i}/{len(queries)} ---")
        print(f"ID: {q['candidate_id']}")
        print(f"Headline: {q['headline']}")
        print(f"Source: {q['source']}")
        print(f"Tickers: {q['tickers']}")
        print()
        print(q['verification_query'])
        print("\n" + "=" * 60 + "\n")


def print_stats() -> None:
    """Print verification statistics."""
    engine = VerificationEngine()
    stats = engine.get_verification_stats()

    print("=== VERIFICATION ENGINE STATS ===\n")
    print(f"Candidates by status: {stats['candidates_by_status']}")
    print(f"Total verified: {stats['total_verified']}")
    print(f"Average confidence: {stats['avg_confidence']}/10")
    print(f"Pushes by status: {stats['pushes_by_status']}")

    if stats['recent_verified']:
        print("\nRecent high-confidence stories:")
        for story in stats['recent_verified']:
            print(f"  [{story['confidence']}/10] {story['summary']}")
