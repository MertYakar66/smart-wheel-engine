"""
News Pipeline Orchestrator (Browser-Based)

Zero API cost architecture using browser automation.

Pipeline:
1. SCRAPE: RSS feeds + optional browser scraping
2. PREPROCESS: Local LLM or rules (filter, categorize)
3. VERIFY: Claude browser session (with web search)
4. FORMAT: ChatGPT browser session (structure content)
5. EDITORIAL: Claude browser session (why it matters)
6. PUBLISH: Save to database, export to dashboard

Fallback chain: Claude → ChatGPT → Gemini
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from news_pipeline.browser_agents import (
    BrowserModelSession,
    ChatGPTBrowserAgent,
    ClaudeBrowserAgent,
    GeminiBrowserAgent,
    ModelType,
)
from news_pipeline.browser_agents.base import SessionManager
from news_pipeline.local_llm import LocalPreprocessor
from news_pipeline.models import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    FormattedStory,
    PipelineResult,
    PipelineStage,
    VerificationResult,
    VerificationStatus,
)
from news_pipeline.scrapers import NewsAggregator, NewsItem

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the pipeline orchestrator."""

    # Verification
    min_confidence: int = 6
    max_stories_per_run: int = 20

    # Model preferences
    verification_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CLAUDE, ModelType.GEMINI]
    )
    formatting_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CHATGPT, ModelType.CLAUDE]
    )
    editorial_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CLAUDE, ModelType.CHATGPT]
    )

    # Browser settings
    headless: bool = True
    browser_data_dir: str = str(Path.home() / ".news_pipeline" / "browser_data")

    # Persistence
    state_dir: str = str(Path.home() / ".news_pipeline" / "state")
    output_dir: str = str(Path.home() / ".news_pipeline" / "output")

    # Local LLM
    use_local_llm: bool = True
    ollama_model: str = "qwen2.5:7b"


class NewsPipelineOrchestrator:
    """
    Main orchestrator for the browser-based news pipeline.

    Uses paid subscriptions via browser - zero API cost.
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize orchestrator."""
        self.config = config or OrchestratorConfig()

        # Create directories
        Path(self.config.state_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Components
        self.aggregator = NewsAggregator(use_rss=True, use_browser=False)
        self.preprocessor = LocalPreprocessor(
            model=self.config.ollama_model,
            use_llm=self.config.use_local_llm,
        )
        self.session_manager = SessionManager(self.config.state_dir)

        # State
        self._current_stage = PipelineStage.DISCOVERY
        self._run_id: str = ""

    async def run(self, request: DiscoveryRequest) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            request: Discovery parameters

        Returns:
            PipelineResult with all processed stories
        """
        self._run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()

        result = PipelineResult(
            run_id=self._run_id,
            started_at=started_at,
            request=request,
        )

        logger.info(f"[Pipeline {self._run_id}] Starting")
        logger.info(f"[Pipeline {self._run_id}] Categories: {request.categories}")
        logger.info(f"[Pipeline {self._run_id}] Time window: {request.time_window}")

        try:
            # Stage 1: Scrape news
            self._current_stage = PipelineStage.DISCOVERY
            logger.info(f"[Pipeline {self._run_id}] Stage 1: Scraping news")
            news_items = await self._scrape_news(request)
            result.discovered_count = len(news_items)
            logger.info(f"[Pipeline {self._run_id}] Scraped {len(news_items)} items")

            if not news_items:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                return result

            # Stage 2: Preprocess (local LLM or rules)
            logger.info(f"[Pipeline {self._run_id}] Stage 2: Preprocessing")
            candidates = await self._preprocess(news_items)
            logger.info(f"[Pipeline {self._run_id}] Kept {len(candidates)} candidates")

            if not candidates:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                return result

            # Stage 3: Verify with Claude (browser)
            self._current_stage = PipelineStage.VERIFICATION
            logger.info(f"[Pipeline {self._run_id}] Stage 3: Verification (browser)")
            verified = await self._verify_batch(candidates)
            passed = [v for v in verified if v.confidence >= self.config.min_confidence]
            result.verified_count = len(passed)
            logger.info(
                f"[Pipeline {self._run_id}] Verified {len(verified)}, "
                f"{len(passed)} passed threshold"
            )

            if not passed:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                return result

            # Stage 4: Format with ChatGPT (browser)
            self._current_stage = PipelineStage.FORMATTING
            logger.info(f"[Pipeline {self._run_id}] Stage 4: Formatting (browser)")
            formatted = await self._format_batch(passed)
            result.formatted_count = len(formatted)
            logger.info(f"[Pipeline {self._run_id}] Formatted {len(formatted)} stories")

            # Stage 5: Editorial with Claude (browser)
            self._current_stage = PipelineStage.EDITORIAL
            logger.info(f"[Pipeline {self._run_id}] Stage 5: Editorial (browser)")
            finalized = await self._finalize_batch(formatted)
            result.finalized_count = len(finalized)
            result.stories = finalized
            logger.info(f"[Pipeline {self._run_id}] Finalized {len(finalized)} stories")

            # Stage 6: Publish
            self._current_stage = PipelineStage.PUBLISHING
            logger.info(f"[Pipeline {self._run_id}] Stage 6: Publishing")
            published = await self._publish(finalized)
            result.published_count = len(published)
            logger.info(f"[Pipeline {self._run_id}] Published {len(published)} stories")

            result.status = "completed"

        except Exception as e:
            logger.error(f"[Pipeline {self._run_id}] Failed at {self._current_stage.value}: {e}")
            result.status = "failed"
            result.errors.append(str(e))

        finally:
            result.completed_at = datetime.utcnow()
            self.session_manager.save_state()

        duration = result.duration_seconds
        logger.info(
            f"[Pipeline {self._run_id}] {result.status.upper()} in {duration:.1f}s | "
            f"Scraped: {result.discovered_count} | "
            f"Verified: {result.verified_count} | "
            f"Published: {result.published_count}"
        )

        return result

    async def _scrape_news(self, request: DiscoveryRequest) -> list[NewsItem]:
        """Stage 1: Scrape news from sources."""
        hours_map = {
            "overnight": 12,
            "last_1h": 1,
            "last_3h": 3,
            "last_6h": 6,
            "today": 24,
            "yesterday": 48,
        }

        return await self.aggregator.fetch_news(
            categories=list(request.categories),
            tickers=list(request.tickers) if request.tickers else None,
            max_items=self.config.max_stories_per_run * 3,  # Extra for filtering
            hours_back=hours_map.get(request.time_window, 12),
        )

    async def _preprocess(self, items: list[NewsItem]) -> list[CandidateStory]:
        """Stage 2: Preprocess with local LLM or rules."""
        filtered = await self.preprocessor.preprocess_batch(
            items,
            filter_threshold=0.3,
        )

        # Convert to CandidateStory
        candidates = []
        for item in filtered:
            candidates.append(
                CandidateStory(
                    story_id=str(uuid.uuid4())[:8],
                    headline=item.headline,
                    source_name=item.source_name,
                    source_url=item.source_url,
                    snippet=item.snippet,
                    tickers=item.tickers,
                    category=item.category.value,
                    source_type=item.source_type.value,
                    discovered_at=datetime.utcnow(),
                )
            )

        return candidates[: self.config.max_stories_per_run]

    async def _verify_batch(
        self,
        candidates: list[CandidateStory],
    ) -> list[VerificationResult]:
        """Stage 3: Verify stories with browser session."""
        results = []

        # Get verification session (prefer Claude for search capability)
        session = await self._get_session(self.config.verification_models)

        if not session:
            logger.error("[Pipeline] No verification session available")
            return results

        for candidate in candidates:
            try:
                # Use Claude's verify method
                if isinstance(session, ClaudeBrowserAgent):
                    result_data = await session.verify_news_story(
                        headline=candidate.headline,
                        source=candidate.source_url,
                        tickers=candidate.tickers,
                        category=candidate.category,
                    )
                elif isinstance(session, GeminiBrowserAgent):
                    result_data = await session.verify_with_search(
                        headline=candidate.headline,
                        source=candidate.source_url,
                        tickers=candidate.tickers,
                        category=candidate.category,
                    )
                else:
                    # Generic prompt
                    result_data = await self._verify_generic(session, candidate)

                if "error" in result_data:
                    logger.warning(f"[Pipeline] Verify failed for {candidate.story_id}")
                    continue

                # Map to VerificationResult
                status_map = {
                    "verified": VerificationStatus.VERIFIED,
                    "partial": VerificationStatus.PARTIAL,
                    "unverified": VerificationStatus.UNVERIFIED,
                    "contradicted": VerificationStatus.CONTRADICTED,
                }

                results.append(
                    VerificationResult(
                        story_id=candidate.story_id,
                        candidate=candidate,
                        status=status_map.get(
                            result_data.get("status", "unverified"),
                            VerificationStatus.UNVERIFIED,
                        ),
                        confidence=result_data.get("confidence", 0),
                        verified_facts=result_data.get("verified_facts", []),
                        what_happened=result_data.get("what_happened", ""),
                        contradictions=result_data.get("contradictions", []),
                        verification_notes=result_data.get("verification_notes", ""),
                    )
                )

            except Exception as e:
                logger.warning(f"[Pipeline] Verify error for {candidate.story_id}: {e}")

        return results

    async def _format_batch(
        self,
        verified: list[VerificationResult],
    ) -> list[FormattedStory]:
        """Stage 4: Format stories with browser session."""
        results = []

        session = await self._get_session(self.config.formatting_models)

        if not session:
            logger.error("[Pipeline] No formatting session available")
            return results

        for v in verified:
            try:
                if isinstance(session, ChatGPTBrowserAgent):
                    result_data = await session.format_story(
                        verified_facts=v.verified_facts,
                        what_happened=v.what_happened,
                        affected_assets=v.candidate.tickers,
                        category=v.candidate.category,
                    )
                else:
                    result_data = await self._format_generic(session, v)

                if "error" in result_data:
                    continue

                results.append(
                    FormattedStory(
                        story_id=v.story_id,
                        title=result_data.get("title", ""),
                        what_happened=result_data.get("what_happened", v.what_happened),
                        bullet_points=result_data.get("bullet_points", v.verified_facts),
                        affected_assets=result_data.get("affected_assets", v.candidate.tickers),
                        category=v.candidate.category,
                        verification_confidence=v.confidence,
                    )
                )

            except Exception as e:
                logger.warning(f"[Pipeline] Format error for {v.story_id}: {e}")

        return results

    async def _finalize_batch(
        self,
        formatted: list[FormattedStory],
    ) -> list[FinalizedStory]:
        """Stage 5: Add editorial polish with browser session."""
        results = []

        session = await self._get_session(self.config.editorial_models)

        if not session:
            logger.error("[Pipeline] No editorial session available")
            return results

        for f in formatted:
            try:
                if isinstance(session, ClaudeBrowserAgent):
                    result_data = await session.add_editorial(
                        title=f.title,
                        what_happened=f.what_happened,
                        bullet_points=f.bullet_points,
                        affected_assets=f.affected_assets,
                        category=f.category,
                    )
                else:
                    result_data = await self._editorial_generic(session, f)

                if "error" in result_data:
                    continue

                results.append(
                    FinalizedStory(
                        story_id=f.story_id,
                        title=result_data.get("title", f.title),
                        what_happened=result_data.get("what_happened", f.what_happened),
                        why_it_matters=result_data.get("why_it_matters", ""),
                        bullet_points=result_data.get("bullet_points", f.bullet_points),
                        affected_assets=result_data.get("affected_assets", f.affected_assets),
                        category=f.category,
                        verification_confidence=f.verification_confidence,
                        market_implications=result_data.get("market_implications"),
                        trading_considerations=result_data.get("trading_considerations"),
                        is_breaking=result_data.get("is_breaking", False),
                        priority=result_data.get("priority", 5),
                        tags=result_data.get("tags", []),
                    )
                )

            except Exception as e:
                logger.warning(f"[Pipeline] Editorial error for {f.story_id}: {e}")

        return results

    async def _publish(self, stories: list[FinalizedStory]) -> list[FinalizedStory]:
        """Stage 6: Save stories to output."""
        # Save to JSON file
        output_file = Path(self.config.output_dir) / f"run_{self._run_id}.json"

        output_data = {
            "run_id": self._run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "stories": [s.to_dict() for s in stories],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"[Pipeline] Saved to {output_file}")

        return stories

    async def _get_session(
        self,
        preferred_models: list[ModelType],
    ) -> BrowserModelSession | None:
        """Get an available browser session with fallback."""
        for model_type in preferred_models:
            session = await self.session_manager.get_session(model_type)
            if session and session.is_ready:
                logger.info(f"[Pipeline] Using {model_type.value} session")
                return session
            elif session and not session.is_authenticated:
                logger.warning(
                    f"[Pipeline] {model_type.value} needs authentication. "
                    f"Run: python morning_run.py --setup-auth"
                )

        return None

    async def _verify_generic(
        self,
        session: BrowserModelSession,
        candidate: CandidateStory,
    ) -> dict:
        """Generic verification prompt."""
        prompt = f"""Verify this financial news:

HEADLINE: "{candidate.headline}"
SOURCE: {candidate.source_url}

Return JSON:
{{"status": "verified/partial/unverified", "confidence": 0-10, "verified_facts": [], "what_happened": ""}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    async def _format_generic(
        self,
        session: BrowserModelSession,
        v: VerificationResult,
    ) -> dict:
        """Generic formatting prompt."""
        prompt = f"""Format this news:

FACTS: {v.verified_facts}
SUMMARY: {v.what_happened}

Return JSON:
{{"title": "", "what_happened": "", "bullet_points": []}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    async def _editorial_generic(
        self,
        session: BrowserModelSession,
        f: FormattedStory,
    ) -> dict:
        """Generic editorial prompt."""
        prompt = f"""Add editorial analysis:

TITLE: {f.title}
WHAT HAPPENED: {f.what_happened}

Return JSON:
{{"why_it_matters": "", "priority": 1-10, "is_breaking": false}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from response."""
        try:
            content = content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "JSON parse failed"}

    async def close(self) -> None:
        """Close all sessions."""
        await self.session_manager.close_all()
        await self.aggregator.close()
