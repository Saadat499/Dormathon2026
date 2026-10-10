"""
Tests for Queue Simulator and Queue Race Engine.
"""

import pytest
from queue_simulator import QueueSimulator, PriorityLevel


def test_queue_simulation_generation():
    """Verify synthetic study stream adheres to data contract."""
    sim = QueueSimulator(arrival_rate_per_hour=15.0, reading_speed_per_hour=18.0, seed=42)
    stream = sim.generate_synthetic_study_stream(duration_hours=2.0)
    assert len(stream) > 10

    first = stream[0]
    assert "study_id" in first
    assert "findings" in first
    assert "priority" in first
    assert "confidence_band" in first
    assert "ood_flag" in first
    assert "read_time_needed" in first


def test_queue_race_stat_speedup():
    """Verify that AI-Triage substantially speeds up reading of STAT cases compared to FIFO."""
    sim = QueueSimulator(arrival_rate_per_hour=18.0, reading_speed_per_hour=20.0, seed=42)
    stream = sim.generate_synthetic_study_stream(duration_hours=6.0)

    race = sim.run_queue_race_comparison(stream=stream)
    fifo_stat_tat = race["fifo"]["by_priority"]["STAT"]["mean_tat_minutes"]
    ai_stat_tat = race["ai_triage"]["by_priority"]["STAT"]["mean_tat_minutes"]

    # AI triage should process STAT cases faster than chronological FIFO
    assert ai_stat_tat < fifo_stat_tat
    assert race["stat_speedup_factor"] > 1.2
    assert race["time_saved_per_stat_case_minutes"] > 0


def test_anti_starvation_aging():
    """Verify anti-starvation aging prevents routine cases from breaching safety caps."""
    sim = QueueSimulator(
        arrival_rate_per_hour=22.0,  # Overloaded queue
        reading_speed_per_hour=20.0,
        anti_starvation_rate=0.10,
        seed=42
    )
    stream = sim.generate_synthetic_study_stream(duration_hours=4.0)
    df_ai = sim.run_simulation(stream, strategy="ai_triage")

    assert not df_ai.empty
    # Routine cases must all be served within reasonable limits
    max_tat = df_ai["turnaround_time_minutes"].max()
    assert max_tat < 720.0  # Within routine SLA cap
