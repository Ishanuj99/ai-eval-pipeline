"""
Streamlit dashboard for the AI Agent Evaluation Pipeline.
Connects to the FastAPI backend via HTTP.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Eval Pipeline",
    page_icon="🔍",
    layout="wide",
)


def get(path: str, **params) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def post(path: str, json_body: dict = None) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🚀 Submit Conversation", "🔎 Conversations", "📋 Evaluations", "💡 Suggestions", "🔬 Meta-Eval"],
)
st.sidebar.divider()
st.sidebar.caption(f"Backend: `{API_BASE}`")


# ─── Dashboard ────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("AI Agent Evaluation Pipeline")
    st.caption("Real-time visibility into agent quality and improvement opportunities")

    stats = get("/evaluations/stats/summary")
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Conversations", stats["total_conversations"])
        col2.metric("Evaluations", stats["total_evaluations"])
        col3.metric("Avg Score", f"{stats['avg_overall_score']:.2%}")
        col4.metric("Pending", stats["pending_conversations"])
        col5.metric("Open Suggestions", stats["open_suggestions"])

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Dimension Scores")
            dims = {
                "Response Quality": stats["avg_response_quality"],
                "Tool Accuracy": stats["avg_tool_accuracy"],
                "Coherence": stats["avg_coherence"],
            }
            fig = go.Figure(go.Bar(
                x=list(dims.keys()),
                y=list(dims.values()),
                marker_color=["#4C78A8", "#F58518", "#54A24B"],
            ))
            fig.update_layout(yaxis_range=[0, 1], height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Recent Evaluations")
            evals = get("/evaluations/", limit=20)
            if evals and evals.get("items"):
                df = pd.DataFrame(evals["items"])
                if "overall_score" in df.columns:
                    fig2 = px.histogram(df, x="overall_score", nbins=20, title="Score Distribution")
                    fig2.update_layout(height=300, margin=dict(t=30))
                    st.plotly_chart(fig2, use_container_width=True)


# ─── Submit Conversation ───────────────────────────────────────────────────────
elif page == "🚀 Submit Conversation":
    st.title("Submit a Conversation")
    st.caption("Ingest a conversation and run the full evaluation pipeline on it.")

    import json as _json
    import uuid as _uuid

    DEMO_SCENARIOS = {
        "Scenario 1 — Tool Regression (bad date format)": {
            "conversation_id": f"demo_tool_{_uuid.uuid4().hex[:6]}",
            "agent_version": "v2.3.1",
            "turns": [
                {"turn_id": 1, "role": "user", "content": "Book me a flight to NYC next week", "timestamp": "2024-01-15T10:30:00Z"},
                {"turn_id": 2, "role": "assistant", "content": "Sure! Let me search for available flights to NYC.",
                 "tool_calls": [{"tool_name": "flight_search", "parameters": {"destination": "NYC", "date_range": "next-week"}, "result": {"status": "success", "flights": ["AA123", "UA456"]}, "latency_ms": 1400}],
                 "timestamp": "2024-01-15T10:30:02Z"},
            ],
            "metadata": {"total_latency_ms": 1400, "mission_completed": True},
        },
        "Scenario 2 — Context Loss (>5 turns)": {
            "conversation_id": f"demo_coherence_{_uuid.uuid4().hex[:6]}",
            "agent_version": "v2.3.1",
            "turns": [
                {"turn_id": 1, "role": "user", "content": "I want a window seat and vegetarian meal on my flight.", "timestamp": "2024-01-15T10:00:00Z"},
                {"turn_id": 2, "role": "assistant", "content": "Got it — window seat and vegetarian meal noted.", "timestamp": "2024-01-15T10:00:02Z"},
                {"turn_id": 3, "role": "user", "content": "Great. Now book the 9am AA123 flight.", "timestamp": "2024-01-15T10:01:00Z"},
                {"turn_id": 4, "role": "assistant", "content": "Booking AA123 at 9am.", "tool_calls": [{"tool_name": "book_flight", "parameters": {"flight_id": "AA123", "time": "09:00"}, "result": {"status": "success"}, "latency_ms": 300}], "timestamp": "2024-01-15T10:01:02Z"},
                {"turn_id": 5, "role": "user", "content": "What seat and meal did you select for me?", "timestamp": "2024-01-15T10:02:00Z"},
                {"turn_id": 6, "role": "assistant", "content": "I booked you on AA123. Would you like to specify a seat preference?", "timestamp": "2024-01-15T10:02:02Z"},
            ],
            "metadata": {"total_latency_ms": 900, "mission_completed": False},
        },
        "Custom (paste your own JSON)": None,
    }

    scenario = st.selectbox("Choose a scenario or enter custom JSON", list(DEMO_SCENARIOS.keys()))

    preset = DEMO_SCENARIOS[scenario]
    if preset is not None:
        default_json = _json.dumps(preset, indent=2)
    else:
        default_json = _json.dumps({
            "conversation_id": f"custom_{_uuid.uuid4().hex[:8]}",
            "agent_version": "v1.0.0",
            "turns": [
                {"turn_id": 1, "role": "user", "content": "Hello, I need help."},
                {"turn_id": 2, "role": "assistant", "content": "Sure, how can I assist you?"},
            ],
            "metadata": {"total_latency_ms": 500, "mission_completed": True},
        }, indent=2)

    payload_str = st.text_area("Conversation JSON", value=default_json, height=320)

    if st.button("🚀 Submit & Evaluate", type="primary"):
        try:
            payload = _json.loads(payload_str)
        except _json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            payload = None

        if payload:
            with st.spinner("Submitting and running evaluation..."):
                result = post("/conversations/", payload)

            if result:
                conv_id = result.get("conversation_id")
                status = result.get("status")
                st.success(f"✅ Conversation `{conv_id}` submitted — status: **{status}**")

                evals = get(f"/evaluations/by-conversation/{conv_id}")
                if evals:
                    ev = evals[0]
                    st.subheader("Evaluation Results")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    scores = ev.get("scores", {})
                    m1.metric("Overall", f"{ev.get('overall_score', 0):.2%}")
                    m2.metric("Quality", f"{scores.get('response_quality', 0):.2%}")
                    m3.metric("Tools", f"{scores.get('tool_accuracy', 0):.2%}")
                    m4.metric("Coherence", f"{scores.get('coherence', 0):.2%}")
                    m5.metric("Heuristic", f"{scores.get('heuristic', 0):.2%}")

                    if ev.get("issues"):
                        st.subheader("Issues Detected")
                        for issue in ev["issues"]:
                            sev = issue.get("severity", "info")
                            icon = "⚠️" if sev == "warning" else "🔴" if sev == "error" else "ℹ️"
                            st.write(f"{icon} **{issue.get('type')}**: {issue.get('description')}")
                    else:
                        st.success("No issues detected.")

                    if ev.get("improvement_suggestions"):
                        st.subheader("Improvement Suggestions")
                        for sug in ev["improvement_suggestions"]:
                            st.info(f"**{sug.get('type', 'prompt').upper()}**: {sug.get('suggestion')} — _{sug.get('rationale', '')}_")

                    with st.expander("Raw evaluation JSON"):
                        st.json(ev)
                else:
                    st.info(f"Evaluation queued. Check **Conversations** page for `{conv_id}`.")


# ─── Conversations ─────────────────────────────────────────────────────────────
elif page == "🔎 Conversations":
    st.title("Conversations")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_id = st.text_input("Conversation ID", placeholder="conv_abc123")
    with col2:
        status_filter = st.selectbox("Status", ["all", "pending", "evaluating", "completed", "failed"])

    if search_id:
        conv = get(f"/conversations/{search_id}")
        if conv:
            st.subheader(f"Conversation: {conv['conversation_id']}")
            st.caption(f"Agent: {conv['agent_version']} | Status: {conv['status']}")

            for turn in conv.get("turns", []):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                with st.chat_message(role):
                    st.write(content)
                    for tc in turn.get("tool_calls", []):
                        with st.expander(f"🔧 Tool: {tc['tool_name']}"):
                            st.json(tc)

            # Show evaluation
            evals = get(f"/evaluations/by-conversation/{search_id}")
            if evals:
                ev = evals[0]
                st.subheader("Latest Evaluation")
                m1, m2, m3, m4 = st.columns(4)
                scores = ev.get("scores", {})
                m1.metric("Overall", f"{ev.get('overall_score', 0):.2%}")
                m2.metric("Quality", f"{scores.get('response_quality', 0):.2%}")
                m3.metric("Tools", f"{scores.get('tool_accuracy', 0):.2%}")
                m4.metric("Coherence", f"{scores.get('coherence', 0):.2%}")

                if ev.get("issues"):
                    st.warning("Issues detected")
                    for issue in ev["issues"]:
                        sev = issue.get("severity", "info")
                        icon = "⚠️" if sev == "warning" else "🔴" if sev == "error" else "ℹ️"
                        st.write(f"{icon} **{issue.get('type')}**: {issue.get('description')}")
    else:
        params = {}
        if status_filter != "all":
            params["status"] = status_filter
        convs = get("/conversations/", **params, limit=50)
        if convs:
            df = pd.DataFrame(convs.get("items", []))
            if not df.empty:
                st.dataframe(df, use_container_width=True)


# ─── Evaluations ───────────────────────────────────────────────────────────────
elif page == "📋 Evaluations":
    st.title("Evaluations")

    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Min Score", 0.0, 1.0, 0.0, 0.05)
    with col2:
        max_score = st.slider("Max Score", 0.0, 1.0, 1.0, 0.05)

    evals = get("/evaluations/", min_score=min_score, max_score=max_score, limit=100)
    if evals:
        items = evals.get("items", [])
        st.caption(f"Showing {len(items)} of {evals.get('total', 0)} evaluations")

        if items:
            df = pd.DataFrame(items)
            # Flatten scores
            scores_df = pd.json_normalize(df["scores"].dropna())
            display_cols = ["evaluation_id", "conversation_id", "overall_score", "evaluator_version", "created_at"]
            display_df = df[[c for c in display_cols if c in df.columns]]
            st.dataframe(display_df, use_container_width=True)

            selected = st.selectbox("View evaluation detail", [""] + [e["evaluation_id"] for e in items])
            if selected:
                ev = next((e for e in items if e["evaluation_id"] == selected), None)
                if ev:
                    st.json(ev)


# ─── Suggestions ───────────────────────────────────────────────────────────────
elif page == "💡 Suggestions":
    st.title("Improvement Suggestions")

    col1, col2 = st.columns([3, 1])
    with col1:
        sug_type = st.selectbox("Type", ["all", "prompt", "tool"])
    with col2:
        if st.button("⚡ Generate Now"):
            result = post("/suggestions/generate/sync")
            if result:
                st.success(f"Created {result.get('suggestions_created', 0)} suggestions")

    params = {}
    if sug_type != "all":
        params["suggestion_type"] = sug_type

    suggestions_data = get("/suggestions/", **params, limit=50)
    if suggestions_data:
        for s in suggestions_data:
            confidence = s.get("confidence") or 0
            status = s.get("status", "pending")
            status_color = "🟡" if status == "pending" else "🟢" if status == "applied" else "🔴"

            with st.expander(
                f"{status_color} [{s['suggestion_type'].upper()}] {s.get('target', 'general')} — confidence: {confidence:.0%}"
            ):
                st.markdown(f"**Suggestion:** {s['suggestion']}")
                if s.get("rationale"):
                    st.markdown(f"**Rationale:** {s['rationale']}")
                if s.get("expected_impact"):
                    st.markdown(f"**Expected Impact:** {s['expected_impact']}")
                if s.get("failure_patterns"):
                    st.markdown(f"**Root Causes:** {', '.join(s['failure_patterns'])}")

                col_a, col_b, _ = st.columns([1, 1, 4])
                if col_a.button("✅ Apply", key=f"apply_{s['suggestion_id']}"):
                    requests.patch(f"{API_BASE}/suggestions/{s['suggestion_id']}/status?status=applied")
                    st.rerun()
                if col_b.button("❌ Reject", key=f"reject_{s['suggestion_id']}"):
                    requests.patch(f"{API_BASE}/suggestions/{s['suggestion_id']}/status?status=rejected")
                    st.rerun()


# ─── Meta-Eval ─────────────────────────────────────────────────────────────────
elif page == "🔬 Meta-Eval":
    st.title("Meta-Evaluation: Evaluator Health")
    st.caption("How well do automated evaluators agree with human annotations?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Recompute Metrics"):
            result = post("/feedback/meta-eval/compute")
            if result:
                st.success(f"Computed: {result.get('computed', {})}")

    metrics = get("/feedback/meta-eval/metrics")
    if metrics:
        df = pd.DataFrame(metrics)
        if not df.empty:
            pivot = df.pivot_table(index="evaluator_name", columns="metric_name", values="value")
            st.subheader("Evaluator Performance Matrix")
            st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)

            st.subheader("Correlation Trend")
            corr_df = df[df["metric_name"] == "correlation"]
            if not corr_df.empty:
                fig = px.bar(corr_df, x="evaluator_name", y="value", color="value",
                             color_continuous_scale="RdYlGn", range_color=[0, 1])
                fig.add_hline(y=0.6, line_dash="dash", annotation_text="Min threshold (0.6)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No meta-eval metrics yet. Add human annotations via the API, then compute metrics.")

    st.divider()
    st.subheader("Evaluator Drift Alerts")
    drift = get("/feedback/meta-eval/drift")
    if drift:
        for alert in drift:
            st.warning(alert["message"])
    else:
        st.success("No drift detected — all evaluators within threshold.")
