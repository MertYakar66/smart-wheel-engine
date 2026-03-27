"""
News Processing Pipeline

Components:
- Entity extraction and tickerization
- Story clustering and deduplication
- Impact scoring
- Brief generation
"""

from .entity_extractor import EntityExtractor
from .story_clusterer import StoryClusterer
from .impact_scorer import ImpactScorer
from .brief_generator import BriefGenerator

__all__ = [
    "EntityExtractor",
    "StoryClusterer",
    "ImpactScorer",
    "BriefGenerator",
]
