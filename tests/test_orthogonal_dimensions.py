import numpy as np
import pytest

from src.compass.orthogonal_dimensions import OrthogonalDimensionsAnalyzer
from src.integration.continuity_field import ContinuityField


def test_prompt_construction():
    analyzer = OrthogonalDimensionsAnalyzer()
    prompt = analyzer.construct_analysis_prompt("Cat", "Dog", "Pet context")
    assert "Concept A: Cat" in prompt
    assert "Concept B: Dog" in prompt
    assert "Context: Pet context" in prompt

def test_vector_analysis_no_field():
    analyzer = OrthogonalDimensionsAnalyzer()
    result = analyzer.analyze_vectors(np.zeros(5), np.zeros(5))
    assert "error" in result

def test_vector_analysis_with_field():
    field = ContinuityField(embedding_dim=2)
    field.add_anchor(np.array([0.0, 0.0]))

    analyzer = OrthogonalDimensionsAnalyzer(continuity_field=field)

    vec_a = np.array([0.1, 0.1]) # Low drift
    vec_b = np.array([1.0, 1.0]) # High drift

    result = analyzer.analyze_vectors(vec_a, vec_b)

    assert "drift_a" in result
    assert "drift_b" in result
    assert result["drift_a"] < result["drift_b"]
