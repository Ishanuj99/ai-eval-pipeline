"""
Seed the pipeline with demo data covering all 3 sample scenarios.
Usage: python scripts/seed_demo.py [--api-url http://localhost:8000]
"""
import argparse
import json
import sys

import requests

DEMO_CONVERSATIONS = [
    # Scenario 1: Tool call regression (wrong date format)
    {
        "conversation_id": "conv_scenario1_001",
        "agent_version": "v2.3.1",
        "turns": [
            {"turn_id": 1, "role": "user", "content": "Book a flight to NYC next week",
             "timestamp": "2024-01-15T10:30:00Z"},
            {"turn_id": 2, "role": "assistant",
             "content": "I'll search for flights to NYC for next week.",
             "tool_calls": [{
                 "tool_name": "flight_search",
                 "parameters": {"destination": "NYC", "date_range": "next week"},  # bad format
                 "result": {"status": "error", "message": "Invalid date format"},
                 "latency_ms": 450,
             }],
             "timestamp": "2024-01-15T10:30:02Z"},
        ],
        "feedback": {
            "user_rating": 2,
            "ops_review": {"quality": "poor", "notes": "Wrong date format in tool call"},
            "annotations": [{"type": "tool_accuracy", "label": "incorrect", "annotator_id": "ann_001"}],
        },
        "metadata": {"total_latency_ms": 1450, "mission_completed": False},
    },
    # Scenario 2: Context loss on long conversation
    {
        "conversation_id": "conv_scenario2_001",
        "agent_version": "v2.3.1",
        "turns": [
            {"turn_id": 1, "role": "user", "content": "I prefer window seats and vegetarian meals"},
            {"turn_id": 2, "role": "assistant", "content": "Got it! I'll remember your preferences."},
            {"turn_id": 3, "role": "user", "content": "Find me a flight to London"},
            {"turn_id": 4, "role": "assistant", "content": "Here are flights to London.",
             "tool_calls": [{"tool_name": "flight_search",
                             "parameters": {"destination": "London"},
                             "result": {"status": "success", "flights": ["BA123", "VS456"]},
                             "latency_ms": 500}]},
            {"turn_id": 5, "role": "user", "content": "Book the first one"},
            {"turn_id": 6, "role": "assistant", "content": "Booking BA123 for you.",
             "tool_calls": [{"tool_name": "flight_book",
                             "parameters": {"flight_id": "BA123", "meal": "standard", "seat": "aisle"},
                             "result": {"status": "success", "booking_id": "BK789"},
                             "latency_ms": 600}]},
        ],
        "feedback": {
            "user_rating": 3,
            "annotations": [
                {"type": "coherence", "label": "incorrect", "annotator_id": "ann_001",
                 "notes": "Agent forgot window seat and vegetarian preferences"},
            ],
        },
        "metadata": {"total_latency_ms": 1100, "mission_completed": True},
    },
    # Scenario 3: Annotator disagreement on helpfulness
    {
        "conversation_id": "conv_scenario3_001",
        "agent_version": "v2.3.1",
        "turns": [
            {"turn_id": 1, "role": "user", "content": "What is the best time to visit Tokyo?"},
            {"turn_id": 2, "role": "assistant",
             "content": "Tokyo is great year-round. Spring (March-May) has cherry blossoms, "
                        "autumn (Sept-Nov) has foliage. Summer can be hot and humid."},
        ],
        "feedback": {
            "user_rating": 4,
            "annotations": [
                {"type": "response_quality", "label": "good", "annotator_id": "ann_001"},
                {"type": "response_quality", "label": "bad", "annotator_id": "ann_002",
                 "notes": "Didn't ask for user's preferences first"},
            ],
        },
        "metadata": {"total_latency_ms": 800, "mission_completed": True},
    },
    # Scenario 4: Good conversation — high quality baseline
    {
        "conversation_id": "conv_good_001",
        "agent_version": "v2.3.1",
        "turns": [
            {"turn_id": 1, "role": "user", "content": "I need to book a flight to NYC next week"},
            {"turn_id": 2, "role": "assistant",
             "content": "I'd be happy to help you book a flight to NYC!",
             "tool_calls": [{
                 "tool_name": "flight_search",
                 "parameters": {"destination": "NYC", "date_range": "2024-01-22/2024-01-28"},
                 "result": {"status": "success", "flights": ["AA100", "DL200"]},
                 "latency_ms": 450,
             }]},
        ],
        "feedback": {
            "user_rating": 4,
            "ops_review": {"quality": "good", "notes": "Correct tool usage"},
            "annotations": [{"type": "tool_accuracy", "label": "correct", "annotator_id": "ann_001"}],
        },
        "metadata": {"total_latency_ms": 1200, "mission_completed": True},
    },
]


def seed(api_url: str):
    print(f"Seeding demo data to {api_url}...")
    for conv in DEMO_CONVERSATIONS:
        try:
            r = requests.post(f"{api_url}/conversations/", json=conv, timeout=10)
            if r.status_code == 202:
                print(f"  ✓ Queued {conv['conversation_id']}")
            elif r.status_code == 409:
                print(f"  ~ Already exists: {conv['conversation_id']}")
            else:
                print(f"  ✗ Failed {conv['conversation_id']}: {r.status_code} {r.text}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Add annotator disagreement for scenario 3
    print("\nAdding cross-annotator annotation for disagreement scenario...")
    try:
        r = requests.post(f"{api_url}/feedback/annotations", json={
            "conversation_id": "conv_scenario3_001",
            "annotator_id": "ann_003",
            "annotation_type": "response_quality",
            "label": "good",
            "confidence": 0.6,
        }, timeout=10)
        print(f"  ✓ Added tiebreaker annotation" if r.ok else f"  ~ {r.status_code}")
    except Exception as e:
        print(f"  ✗ {e}")

    print("\nDone! Wait a few seconds for workers to process evaluations, then visit:")
    print(f"  API docs:  {api_url}/docs")
    print(f"  Dashboard: http://localhost:8501")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    seed(args.api_url)
