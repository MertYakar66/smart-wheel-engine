"""
Processing Pipeline for Macro + SP500 Event Intelligence System

Components:
- ArticleClassifier: Rule-based category classification
- StoryClusterer: Group articles into stories
- StoryRanker: Score by macro/SP500 relevance
- BriefGenerator: Morning/Evening book generation

Legacy components (for compatibility):
- EntityExtractor, ImpactScorer
"""

from .brief_generator import BriefGenerator
from .classifier import ArticleClassifier
from .clusterer import StoryClustering

# Legacy imports for compatibility
from .entity_extractor import EntityExtractor
from .impact_scorer import ImpactScorer
from .ranker import StoryRanker
from .story_clusterer import StoryClusterer

__all__ = [
    # New components
    "ArticleClassifier",
    "StoryClustering",
    "StoryRanker",
    "BriefGenerator",
    # Legacy
    "EntityExtractor",
    "StoryClusterer",
    "ImpactScorer",
]
