"""Unit tests for Sovereign Manifold Verifier and Harness Logic."""

import pytest
from sovereign_manifold import SovereignManifoldVerifier


def test_ple_scoring_formula_ideal():
    # p = 1.0, G = 1.0, O' = 1.0 -> s = 0.4*1.0 + 0.6*1.0*1.0 = 1.0
    res = SovereignManifoldVerifier.compute_ple_score(
        constraints_total=10,
        constraints_met=10,
        hard_constraints_violated=0,
        o_achieved=1.0,
        o_naive=0.0,
    )
    assert res["score"] == 1.0
    assert res["G"] == 1.0
    assert res["p"] == 1.0
    assert res["O_prime"] == 1.0


def test_ple_scoring_hard_constraint_failure():
    # If G=0 (hard constraint broken), max score is capped at 0.4
    res = SovereignManifoldVerifier.compute_ple_score(
        constraints_total=10,
        constraints_met=9,
        hard_constraints_violated=1,
        o_achieved=1.0,
        o_naive=0.0,
    )
    assert res["G"] == 0.0
    assert res["p"] == 0.9
    assert res["score"] == pytest.approx(0.36)


def test_ple_scoring_target_band():
    # Verify score reaches 0.85+ when G=1 and high O'
    res = SovereignManifoldVerifier.compute_ple_score(
        constraints_total=20,
        constraints_met=20,
        hard_constraints_violated=0,
        o_achieved=0.9,
        o_naive=0.1,
    )
    assert res["G"] == 1.0
    assert res["p"] == 1.0
    # O' = (0.9 - 0.1) / (1 - 0.1) = 0.8 / 0.9 = 0.8888...
    # s = 0.4 + 0.6 * 0.8888... = 0.4 + 0.5333... = 0.9333...
    assert res["score"] >= 0.80
    assert res["score"] <= 1.0


def test_aabb_clash_detection():
    # Box 1: [0, 10] in all axes
    b1 = {"min_x": 0.0, "max_x": 10.0, "min_y": 0.0, "max_y": 10.0, "min_z": 0.0, "max_z": 10.0}
    # Box 2: [5, 15] in all axes -> overlaps with Box 1
    b2 = {"min_x": 5.0, "max_x": 15.0, "min_y": 5.0, "max_y": 15.0, "min_z": 5.0, "max_z": 15.0}
    # Box 3: [20, 30] -> disjoint
    b3 = {"min_x": 20.0, "max_x": 30.0, "min_y": 20.0, "max_y": 30.0, "min_z": 20.0, "max_z": 30.0}

    clashes = SovereignManifoldVerifier.detect_3d_aabb_clashes([b1, b2, b3])
    assert clashes == [(0, 1)]


def test_deliverable_auditing():
    paths = [
        "/app/files/solution.py",
        "/filesystem/submission.zip",
        "/app/files/model.ifc",
        "/app/files/temp.tmp",
    ]
    res = SovereignManifoldVerifier.extract_and_audit_deliverables("dummy task", paths)
    assert res["status"] == "VALID"
    assert res["deliverables_count"] == 3
    assert "/app/files/temp.tmp" not in res["deliverables_found"]
