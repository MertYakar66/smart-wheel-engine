"""
Streamlit Dashboard for Financial News Platform

A professional, Bloomberg-style news interface with:
- Category-based navigation
- Story cards with summaries
- Impact scoring indicators
- Watchlist integration
- Brief generation
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    st = None

from financial_news.models import Story
from financial_news.pipeline import NewsPipeline
from financial_news.storage import NewsStore


class NewsDashboard:
    """
    Streamlit-based dashboard for the financial news platform.

    Features:
    - Category tabs
    - Story cards with impact indicators
    - Watchlist management
    - Brief viewing
    - Pipeline controls
    """

    def __init__(self, db_path: str = "data/news_store.db"):
        """Initialize dashboard with database connection"""
        self.store = NewsStore(db_path)
        self.pipeline = NewsPipeline(db_path)

    def run(self) -> None:
        """Run the Streamlit dashboard"""
        if st is None:
            print("Streamlit not installed. Run: pip install streamlit")
            return

        # Page config
        st.set_page_config(
            page_title="Financial News",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS
        st.markdown("""
        <style>
        .story-card {
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #3B82F6;
        }
        .story-card.high-impact {
            border-left-color: #EF4444;
        }
        .story-card.medium-impact {
            border-left-color: #F59E0B;
        }
        .ticker-badge {
            background-color: #374151;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 4px;
        }
        .impact-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .impact-high { background-color: #DC2626; color: white; }
        .impact-medium { background-color: #D97706; color: white; }
        .impact-low { background-color: #059669; color: white; }
        .source-count {
            color: #9CA3AF;
            font-size: 12px;
        }
        .time-ago {
            color: #6B7280;
            font-size: 12px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Sidebar
        self._render_sidebar()

        # Main content
        st.title("Financial News Dashboard")

        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "News Feed", "Categories", "Brief", "Settings"
        ])

        with tab1:
            self._render_news_feed()

        with tab2:
            self._render_categories()

        with tab3:
            self._render_brief()

        with tab4:
            self._render_settings()

    def _render_sidebar(self) -> None:
        """Render sidebar with navigation and stats"""
        with st.sidebar:
            st.header("Navigation")

            # Category filter
            categories = self.store.get_all_categories()
            category_names = ["All"] + [c.name for c in categories]
            selected_category = st.selectbox("Category", category_names)

            if selected_category != "All":
                st.session_state["selected_category"] = selected_category

            # Watchlist
            st.subheader("Watchlist")
            watchlist_input = st.text_input("Add ticker", placeholder="e.g., AAPL")
            if watchlist_input:
                if "watchlist" not in st.session_state:
                    st.session_state["watchlist"] = []
                if watchlist_input.upper() not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(watchlist_input.upper())

            if "watchlist" in st.session_state and st.session_state["watchlist"]:
                st.write("Watching:", ", ".join(st.session_state["watchlist"]))

            # Stats
            st.subheader("Stats")
            stats = self.store.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stories", stats.get("stories", 0))
            with col2:
                st.metric("Articles", stats.get("articles", 0))

            # Pipeline controls
            st.subheader("Pipeline")
            if st.button("Refresh News"):
                with st.spinner("Fetching news..."):
                    asyncio.run(self.pipeline.run_full_pipeline(hours_lookback=6))
                st.success("News refreshed!")
                st.rerun()

    def _render_news_feed(self) -> None:
        """Render the main news feed"""
        st.subheader("Latest Stories")

        # Get stories
        stories = self.store.get_recent_stories(hours=48, limit=30)

        if not stories:
            st.info("No stories available. Click 'Refresh News' to fetch latest.")
            return

        # Filter by category if selected
        selected_cat = st.session_state.get("selected_category")
        if selected_cat:
            categories = self.store.get_all_categories()
            cat = next((c for c in categories if c.name == selected_cat), None)
            if cat:
                topic_values = [t.value for t in cat.topics]
                stories = [s for s in stories if any(
                    t.value in topic_values for t in s.topics
                )]

        # Render story cards
        for story in stories:
            self._render_story_card(story)

    def _render_story_card(self, story: Story) -> None:
        """Render a single story card"""
        # Determine impact level
        if story.impact_score >= 0.7:
            impact_class = "high-impact"
            impact_badge = '<span class="impact-badge impact-high">HIGH</span>'
        elif story.impact_score >= 0.4:
            impact_class = "medium-impact"
            impact_badge = '<span class="impact-badge impact-medium">MED</span>'
        else:
            impact_class = ""
            impact_badge = '<span class="impact-badge impact-low">LOW</span>'

        # Time ago
        time_diff = datetime.utcnow() - story.last_updated_at
        if time_diff.total_seconds() < 3600:
            time_ago = f"{int(time_diff.total_seconds() / 60)}m ago"
        elif time_diff.total_seconds() < 86400:
            time_ago = f"{int(time_diff.total_seconds() / 3600)}h ago"
        else:
            time_ago = f"{int(time_diff.total_seconds() / 86400)}d ago"

        # Ticker badges
        ticker_html = " ".join(
            f'<span class="ticker-badge">${t}</span>'
            for t in story.tickers[:5]
        )

        # Card HTML
        card_html = f"""
        <div class="story-card {impact_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <h4 style="margin: 0; font-size: 16px;">{story.headline}</h4>
                {impact_badge}
            </div>
            <p style="margin: 8px 0; color: #D1D5DB; font-size: 14px;">
                {story.summary or 'No summary available.'}
            </p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>{ticker_html}</div>
                <div>
                    <span class="source-count">{story.source_count} sources</span>
                    <span class="time-ago" style="margin-left: 12px;">{time_ago}</span>
                </div>
            </div>
            {f'<p style="margin-top: 8px; color: #9CA3AF; font-size: 13px; font-style: italic;">{story.why_it_matters}</p>' if story.why_it_matters else ''}
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)

        # Expand for details
        with st.expander("Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Topics:**", ", ".join(t.value for t in story.topics))
                st.write("**Entities:**", ", ".join(e.name for e in story.entities[:5]))
            with col2:
                st.write("**Impact Score:**", f"{story.impact_score:.2f}")
                st.write("**Confidence:**", f"{story.confidence_score:.2f}")
                st.write("**Articles:**", len(story.article_ids))

    def _render_categories(self) -> None:
        """Render category management"""
        st.subheader("Categories")

        categories = self.store.get_all_categories()

        for category in categories:
            with st.expander(f"{category.name} - {category.description}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Keywords:**", ", ".join(category.keywords[:10]))
                    st.write("**Topics:**", ", ".join(t.value for t in category.topics))

                with col2:
                    st.write("**Tickers:**", ", ".join(category.tickers[:10]) or "All")
                    st.write("**Last Fetch:**", category.last_successful_fetch or "Never")

                # Get stories for this category
                stories = []
                for topic in category.topics:
                    topic_stories = self.store.get_stories_by_topic(topic, limit=5)
                    stories.extend(topic_stories)

                if stories:
                    st.write(f"**Recent Stories:** {len(stories)}")

    def _render_brief(self) -> None:
        """Render morning/evening brief"""
        st.subheader("Your Brief")

        brief_type = st.radio(
            "Brief Type",
            ["Morning", "Evening"],
            horizontal=True,
        )

        # Get recent stories for brief
        stories = self.store.get_recent_stories(hours=12, limit=15)

        if not stories:
            st.info("No stories available for brief.")
            return

        # Generate brief display
        st.markdown("---")

        # Executive summary
        st.markdown("### Executive Summary")
        top_stories = sorted(stories, key=lambda s: s.impact_score, reverse=True)[:3]
        summary_points = [s.headline for s in top_stories]
        for point in summary_points:
            st.markdown(f"- {point}")

        # Top stories
        st.markdown("### Top Stories")
        for i, story in enumerate(top_stories[:5], 1):
            st.markdown(f"**{i}. {story.headline}**")
            st.markdown(f"_{story.summary or 'No summary'}_")
            if story.why_it_matters:
                st.markdown(f"*Why it matters:* {story.why_it_matters}")
            st.markdown("")

        # Export button
        if st.button("Export Brief"):
            brief_text = self._generate_brief_text(stories[:10], brief_type.lower())
            st.download_button(
                "Download",
                brief_text,
                f"brief_{datetime.now().strftime('%Y%m%d')}.txt",
            )

    def _generate_brief_text(self, stories: list[Story], brief_type: str) -> str:
        """Generate text version of brief"""
        lines = [
            f"MARKET BRIEF - {brief_type.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50,
            "",
            "TOP STORIES",
            "-" * 20,
        ]

        for i, story in enumerate(stories, 1):
            lines.extend([
                f"{i}. {story.headline}",
                f"   {story.summary or ''}",
                f"   Tickers: {', '.join(story.tickers[:5]) or 'N/A'}",
                f"   Impact: {'HIGH' if story.impact_score > 0.7 else 'MEDIUM' if story.impact_score > 0.4 else 'LOW'}",
                "",
            ])

        return "\n".join(lines)

    def _render_settings(self) -> None:
        """Render settings page"""
        st.subheader("Settings")

        # User preferences
        st.markdown("### Preferences")

        st.selectbox(
            "Timezone",
            ["America/Toronto", "America/New_York", "America/Los_Angeles", "Europe/London", "Asia/Tokyo"],
        )

        st.time_input("Morning Brief Time", datetime.strptime("07:00", "%H:%M"))
        st.time_input("Evening Brief Time", datetime.strptime("19:00", "%H:%M"))

        # Notifications
        st.markdown("### Notifications")
        st.checkbox("Email Digest", value=True)
        st.checkbox("Push Notifications", value=False)

        # Pipeline settings
        st.markdown("### Pipeline")
        st.slider("Hours Lookback", 1, 48, 12)

        if st.button("Save Settings"):
            st.success("Settings saved!")

        # Database management
        st.markdown("### Database")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clean Old Data"):
                deleted = self.store.cleanup_old_data(max_age_days=30)
                st.success(f"Cleaned: {deleted}")

        with col2:
            stats = self.store.get_stats()
            st.json(stats)


def main():
    """Run the dashboard"""
    dashboard = NewsDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
