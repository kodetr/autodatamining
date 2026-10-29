"""Tests for interactive glossary (explain_concept and list_concepts)."""

import pytest

from autodatamining import explain_concept, list_concepts
from autodatamining.exceptions import DataValidationError


def test_list_concepts():
    concepts = list_concepts()
    assert isinstance(concepts, list)
    assert "overfitting" in concepts
    assert "cross_validation" in concepts
    assert "lift" in concepts
    assert "silhouette" in concepts


def test_explain_overfitting():
    text = explain_concept("overfitting")
    assert "KONSEP DATA MINING: OVERFITTING" in text
    assert "Definisi" in text
    assert "Analogi" in text
    assert "Tips Praktis" in text


def test_explain_concept_aliases():
    text_cv = explain_concept("cv")
    assert "CROSS-VALIDATION" in text_cv

    text_prec = explain_concept("precision")
    assert "PRESISI VS RECALL" in text_prec


def test_invalid_concept():
    with pytest.raises(DataValidationError):
        explain_concept("quantum_mechanics_xyz")
