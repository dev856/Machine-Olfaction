"""Unit tests for semantic odor knowledge base and chemical volatile metadata."""

import pytest
from src.data.semantic_profiles import SemanticKnowledgeBase, get_knowledge_base, get_semantic_profile


def test_knowledge_base_initialization():
    kb = get_knowledge_base()
    all_classes = kb.list_all_classes()
    assert len(all_classes) >= 50
    assert "cinnamon" in all_classes
    assert "banana" in all_classes
    assert "lemon" in all_classes


def test_semantic_profile_fields():
    prof = get_semantic_profile("cinnamon")
    assert prof.name == "cinnamon"
    assert prof.category == "Warm Spices"
    assert "Cinnamaldehyde" in prof.volatiles
    assert len(prof.sensory_notes) > 0
    assert len(prof.full_description) > 10

    d = prof.to_dict()
    assert d["name"] == "cinnamon"
    assert d["category"] == "Warm Spices"


def test_fallback_profile_for_unknown_smell():
    prof = get_semantic_profile("unknown_exotic_flower_xyz")
    assert prof.name == "unknown_exotic_flower_xyz"
    assert prof.category in ["General Odorant", "Uncategorized"]
    assert len(prof.volatiles) > 0


def test_category_filtering():
    kb = get_knowledge_base()
    fruits = kb.get_by_category("Fruits & Citrus")
    assert len(fruits) >= 10
    fruit_names = [f.name for f in fruits]
    assert "apple" in fruit_names
    assert "banana" in fruit_names
    assert "lemon" in fruit_names
