"""
Brief Generator - AI-powered news summaries

Generates:
- Story summaries ("What happened")
- Impact analysis ("Why it matters")
- Morning/Evening briefs with macro sections
- Executive summaries

Uses:
- Local LLM for routine summaries (fast, cost-free)
- Cloud LLM for top stories (higher quality)
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from financial_news.schema import (
    Article,
    Brief,
    CategoryType,
    Story,
)

# Legacy import for compatibility
try:
    from financial_news.models import TopicCategory, UserProfile
except ImportError:
    TopicCategory = CategoryType  # Fallback
    UserProfile = None

logger = logging.getLogger(__name__)


# Prompt templates
STORY_SUMMARY_PROMPT = """You are a financial news analyst. Summarize this news story concisely.

HEADLINE: {headline}

ARTICLES:
{articles}

ENTITIES: {entities}
TICKERS: {tickers}

Generate a JSON response with:
{{
    "summary": "1-2 sentence summary of what happened",
    "why_it_matters": "1-2 sentence explanation of market/economic impact",
    "key_facts": ["fact 1", "fact 2", "fact 3"]
}}

Be specific about numbers, dates, and names. Focus on market-relevant information."""


BRIEF_PROMPT = """You are a senior financial analyst preparing a {brief_type} market brief.

TOP STORIES:
{stories}

USER INTERESTS: {interests}

Generate a professional brief with:
1. Executive Summary (3-4 sentences covering the most important developments)
2. Market Outlook (1-2 sentences on what to watch)

Format as JSON:
{{
    "executive_summary": "...",
    "market_outlook": "..."
}}

Be concise, professional, and actionable."""


class BriefGenerator:
    """
    Generates AI-powered news summaries and briefs.

    Implements tiered approach:
    - Local LLM: All stories (fast, free)
    - Cloud LLM: Top stories only (higher quality)
    """

    def __init__(
        self,
        local_agent: Any | None = None,
        cloud_client: Any | None = None,
        use_cloud_for_top_n: int = 5,
    ):
        """
        Initialize brief generator.

        Args:
            local_agent: Local LLM agent (from local_agent module)
            cloud_client: Optional cloud API client (Claude, etc.)
            use_cloud_for_top_n: Number of top stories to process with cloud
        """
        self.local_agent = local_agent
        self.cloud_client = cloud_client
        self.use_cloud_for_top_n = use_cloud_for_top_n

        # Cache for generated summaries
        self._summary_cache: dict[str, dict[str, str]] = {}

    async def generate_story_summary(
        self,
        story: Story,
        articles: list[Article],
        use_cloud: bool = False,
    ) -> Story:
        """
        Generate summary and impact analysis for a story.

        Args:
            story: Story to summarize
            articles: Articles in the story
            use_cloud: Whether to use cloud LLM (higher quality)

        Returns:
            Story with populated summary and why_it_matters
        """
        # Check cache
        if story.story_id in self._summary_cache:
            cached = self._summary_cache[story.story_id]
            story.summary = cached.get("summary", "")
            story.why_it_matters = cached.get("why_it_matters", "")
            return story

        # Build prompt context
        article_texts = "\n".join([
            f"- {a.title} ({a.source_name}): {a.snippet or ''}"
            for a in articles[:5]  # Limit to 5 articles
        ])

        entities = ", ".join([e.name for e in story.entities[:10]])
        tickers = ", ".join(story.tickers[:10])

        prompt = STORY_SUMMARY_PROMPT.format(
            headline=story.headline,
            articles=article_texts,
            entities=entities,
            tickers=tickers,
        )

        try:
            if use_cloud and self.cloud_client:
                response = await self._generate_cloud(prompt)
            elif self.local_agent:
                response = await self._generate_local(prompt)
            else:
                # Fallback: rule-based summary
                response = self._generate_fallback_summary(story, articles)

            # Parse response
            if isinstance(response, str):
                # Try to parse as JSON
                response = self._parse_json_response(response)

            story.summary = response.get("summary", story.headline)
            story.why_it_matters = response.get(
                "why_it_matters",
                self._generate_default_impact(story)
            )

            # Cache result
            self._summary_cache[story.story_id] = {
                "summary": story.summary,
                "why_it_matters": story.why_it_matters,
            }

        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            story.summary = story.headline
            story.why_it_matters = self._generate_default_impact(story)

        return story

    async def generate_brief(
        self,
        user: UserProfile,
        stories: list[Story],
        brief_type: str = "morning",
    ) -> Brief:
        """
        Generate a morning or evening brief for a user.

        Args:
            user: User profile with interests
            stories: Ranked stories for the brief
            brief_type: "morning" or "evening"

        Returns:
            Generated Brief object
        """
        brief_id = f"{user.user_id}_{brief_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

        # Limit stories
        brief_stories = stories[:user.category_ids.__len__() if user.category_ids else 10]

        # Generate summaries for stories that don't have them
        tasks = []
        for story in brief_stories:
            if not story.summary:
                # Use cloud for top stories
                use_cloud = brief_stories.index(story) < self.use_cloud_for_top_n
                tasks.append(self.generate_story_summary(story, [], use_cloud))

        if tasks:
            await asyncio.gather(*tasks)

        # Count new vs updated
        new_count = sum(
            1 for s in brief_stories
            if s.story_id not in user.last_seen_story_ids
        )
        updated_count = sum(
            1 for s in brief_stories
            if s.story_id in user.last_seen_story_ids and s.previous_summary
        )

        # Generate executive summary
        executive_summary, market_outlook = await self._generate_brief_summary(
            brief_stories,
            user,
            brief_type,
        )

        brief = Brief(
            brief_id=brief_id,
            user_id=user.user_id,
            brief_type=brief_type,
            generated_at=datetime.utcnow(),
            stories=brief_stories,
            new_stories_count=new_count,
            updated_stories_count=updated_count,
            executive_summary=executive_summary,
            market_outlook=market_outlook,
        )

        return brief

    async def _generate_brief_summary(
        self,
        stories: list[Story],
        user: UserProfile,
        brief_type: str,
    ) -> tuple:
        """Generate executive summary and market outlook"""
        if not stories:
            return "No significant news to report.", "Markets quiet."

        # Build story summaries
        story_texts = "\n".join([
            f"- {s.headline}: {s.summary}"
            for s in stories[:10]
        ])

        interests = ", ".join(user.watchlist_tickers[:10])

        prompt = BRIEF_PROMPT.format(
            brief_type=brief_type,
            stories=story_texts,
            interests=interests or "general markets",
        )

        try:
            if self.cloud_client:
                response = await self._generate_cloud(prompt)
            elif self.local_agent:
                response = await self._generate_local(prompt)
            else:
                return self._generate_fallback_brief(stories)

            if isinstance(response, str):
                response = self._parse_json_response(response)

            return (
                response.get("executive_summary", ""),
                response.get("market_outlook", ""),
            )

        except Exception as e:
            logger.warning(f"Brief generation failed: {e}")
            return self._generate_fallback_brief(stories)

    async def _generate_local(self, prompt: str) -> dict[str, Any]:
        """Generate using local LLM"""
        if not self.local_agent:
            raise ValueError("No local agent configured")

        response = await self.local_agent.generate(
            prompt=prompt,
            temperature=0.3,
            json_mode=True,
        )

        return self._parse_json_response(response)

    async def _generate_cloud(self, prompt: str) -> dict[str, Any]:
        """Generate using cloud LLM (Claude)"""
        if not self.cloud_client:
            raise ValueError("No cloud client configured")

        # Assume Claude-style API
        response = await self.cloud_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        return self._parse_json_response(text)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response"""
        import re

        # Clean up response
        text = text.strip()

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Find JSON object
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"summary": text, "why_it_matters": ""}

    def _generate_fallback_summary(
        self,
        story: Story,
        articles: list[Article],
    ) -> dict[str, str]:
        """Rule-based summary fallback"""
        summary = story.headline

        # Add source count
        if story.source_count > 1:
            summary += f" (reported by {story.source_count} sources)"

        return {
            "summary": summary,
            "why_it_matters": self._generate_default_impact(story),
        }

    def _generate_default_impact(self, story: Story) -> str:
        """Generate default impact statement based on categories and factors"""
        # Map factors to impact statements
        factor_impacts = {
            "rates": "May affect interest rate expectations and bond yields.",
            "inflation": "Could impact inflation outlook and Fed policy.",
            "growth": "May signal shifts in economic growth trajectory.",
            "oil": "Could affect energy costs and inflation.",
            "risk": "May shift risk sentiment across markets.",
            "earnings": "Could impact stock price and sector sentiment.",
            "consumer": "May reflect consumer spending trends.",
            "defense": "Could affect defense sector and risk premia.",
        }

        # Check affected factors
        for factor in story.affected_factors:
            factor_lower = factor.lower()
            if factor_lower in factor_impacts:
                return factor_impacts[factor_lower]

        # Check categories
        category_impacts = {
            CategoryType.FED_RATES.value: "May affect interest rate expectations and bond yields.",
            CategoryType.INFLATION.value: "Could impact inflation outlook and Fed policy.",
            CategoryType.LABOR.value: "May signal labor market strength or weakness.",
            CategoryType.GROWTH_CONSUMER.value: "Could affect growth and consumer outlook.",
            CategoryType.OIL_ENERGY.value: "May impact energy costs and inflation.",
            CategoryType.GEOPOLITICS.value: "Could affect global markets and risk sentiment.",
            CategoryType.SP500_CORPORATE.value: "Could impact stock price and sector sentiment.",
        }

        for cat_id in story.category_scores:
            if cat_id in category_impacts:
                return category_impacts[cat_id]

        if story.tickers:
            return f"May affect {', '.join(story.tickers[:3])} and related securities."

        return "Monitor for further developments."

    def _generate_fallback_brief(self, stories: list[Story]) -> tuple:
        """Generate fallback brief without LLM"""
        if not stories:
            return "No significant developments.", "Markets stable."

        top_story = stories[0]
        exec_summary = f"Top story: {top_story.headline}"

        if len(stories) > 1:
            exec_summary += f" Also notable: {stories[1].headline}"

        outlook = "Watch for earnings and economic data releases."

        return exec_summary, outlook

    def clear_cache(self) -> None:
        """Clear summary cache"""
        self._summary_cache.clear()
