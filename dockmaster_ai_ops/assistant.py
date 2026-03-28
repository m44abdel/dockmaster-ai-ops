"""
Grounded DockMaster AI Ops Assistant — answers are derived only from live KPIs,
work orders, schedule output, technician roster, and baseline comparison dicts.
Optional Gemini pass rephrases a fixed draft; facts are computed in Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DOTENV_LOADED = False


def _manual_load_env_file(env_path: Path) -> None:
    """Load ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` without python-dotenv (fallback)."""
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and val:
            os.environ[key] = val


def ensure_dotenv_loaded(project_root: Path | None = None) -> None:
    """
    Load ``.env`` from the project root (folder that contains ``app.py``).

    Uses ``python-dotenv`` when installed; always falls back to a small parser
    if the API key is still missing (common when ``dotenv`` is not installed).
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    root = project_root or Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=True)
    except ImportError:
        pass
    if not (os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()):
        _manual_load_env_file(env_path)


def _load_dotenv_from_project_root() -> None:
    """Backward-compatible alias for :func:`ensure_dotenv_loaded`."""
    ensure_dotenv_loaded()

# Urgent threshold aligned with product copy / examples
URGENT_URGENCY_MIN = 70.0
HIGH_RISK_MIN = 0.75
EARLY_SLOT_MAX = 5


def _parts_ready(row: pd.Series) -> bool:
    eta = int(row.get("parts_eta_slot", 0) or 0)
    return eta <= 0


def _priority_col(wo: pd.DataFrame) -> str:
    return (
        "scheduling_priority_score"
        if "scheduling_priority_score" in wo.columns
        else "urgency_score"
    )


def build_assistant_context(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    techs: pd.DataFrame,
    kpis: dict[str, Any],
    baseline_comparison: dict[str, Any],
    horizon_slots: float,
    scenario_name: str,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured snapshot for the assistant (LLM or deterministic tools)."""
    wo_records = wo.replace({np.nan: None}).to_dict(orient="records")
    sch = sched.replace({np.nan: None}).to_dict(orient="records") if len(sched) else []
    tch = techs.replace({np.nan: None}).to_dict(orient="records")
    out: dict[str, Any] = {
        "scenario": scenario_name,
        "horizon_slots": float(horizon_slots),
        "kpis": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in kpis.items()},
        "schedule": sch,
        "work_orders": wo_records,
        "technicians": tch,
        "baseline_comparison": baseline_comparison,
    }
    if run_metadata:
        out["run_metadata"] = run_metadata
    return out


def _context_json_for_llm(context: dict[str, Any]) -> str:
    return json.dumps(context, indent=2, default=str)


def answer_question_from_context(
    user_question: str,
    context: dict[str, Any],
    api_key: str,
    *,
    chat_history: list[dict[str, str]] | None = None,
    model_name: str | None = None,
) -> str:
    """
    Answer a free-form question using only the provided context (Gemini).
    ``chat_history`` entries: ``{"role": "user"|"assistant", "content": "..."}`` for short follow-ups.
    """
    if not api_key or not str(api_key).strip():
        return "Configure GEMINI_API_KEY in `.env` to use the assistant."
    # Default avoids gemini-2.0-flash (restricted for new API keys as of 2025+).
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    ctx_text = _context_json_for_llm(context)

    history_lines: list[str] = []
    if chat_history:
        recent = chat_history[-8:]  # last 4 turns max
        for m in recent:
            r = m.get("role", "")
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if r == "user":
                history_lines.append(f"User (earlier): {c}")
            elif r == "assistant":
                history_lines.append(f"Assistant (earlier): {c}")
    history_block = "\n".join(history_lines) if history_lines else "(no prior messages)"

    prompt = f"""You are **DockMaster AI Ops Assistant** — an operations copilot for the *current* demo plan only.

## Rules (must follow)
- Base your answer **only** on the JSON context below: `run_metadata`, `kpis`, `work_orders`, `schedule`, `technicians`, `baseline_comparison`, `scenario`, `horizon_slots`.
- Do **not** invent work order IDs, technicians, numbers, or events that are not in the context.
- If the question cannot be fully answered from the context, say what you **can** answer and what is **not** in the data.
- Use operational language (clear, concise). Use bullet lists when comparing multiple jobs or techs.
- **Parts:** `parts_eta_slot` is the first slot index when parts are available; `0` means ready at the start.
- **Schedule:** rows link `work_order_id` to `assigned_technician`, `start_slot`, `end_slot`, `lateness_slots` (vs promised date).
- **Priorities:** use `scheduling_priority_score`, `urgency_score`, `failure_risk` from work orders as given.
- Do not imply the LLM runs the optimizer or the risk model; you are explaining **this run’s** outputs.

## Prior conversation (for follow-up questions only; facts still come from JSON)
{history_block}

## Full grounded context (JSON)
```json
{ctx_text}
```

## User question
{user_question.strip()}

## Your answer
"""

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key.strip())
        mdl = genai.GenerativeModel(model)
        resp = mdl.generate_content(
            prompt,
            generation_config={
                "temperature": 0.25,
                "max_output_tokens": 4096,
            },
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text and getattr(resp, "candidates", None):
            try:
                parts = resp.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
            except (IndexError, AttributeError, ValueError):
                text = ""
        return text if text else "No response text returned (blocked or empty). Rephrase or check API quota."
    except ModuleNotFoundError as e:
        if "generativeai" in str(e) or "google" in str(e):
            return (
                "**Missing package:** `google-generativeai` is not installed in the Python environment "
                "running Streamlit.\n\n"
                "In **that same environment**, run:\n\n"
                "`pip install google-generativeai`\n\n"
                "or `pip install -r requirements.txt`, then restart Streamlit."
            )
        return f"Assistant error (ModuleNotFoundError): {e!s}."
    except Exception as e:
        return f"Assistant error ({type(e).__name__}): {e!s}. Check API key and model name."


def explain_work_order(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    work_order_id: str,
) -> dict[str, Any]:
    row = wo.loc[wo["work_order_id"] == work_order_id]
    if row.empty:
        return {
            "title": "Work order not found",
            "facts": {},
            "answer": f"No work order `{work_order_id}` in the current plan data.",
        }
    wo_row = row.iloc[0]
    sch_row = sched.loc[sched["work_order_id"] == work_order_id] if len(sched) else pd.DataFrame()
    on_schedule = not sch_row.empty
    facts: dict[str, Any] = {
        "failure_risk": float(wo_row["failure_risk"]),
        "urgency_score": float(wo_row["urgency_score"]),
        "parts_ready_now": _parts_ready(wo_row),
        "parts_eta_slot": int(wo_row.get("parts_eta_slot", 0) or 0),
        "required_skill": str(wo_row["required_skill"]),
        "required_skill_secondary": str(wo_row.get("required_skill_secondary", "") or ""),
        "required_bay_type": str(wo_row["required_bay_type"]),
        "promised_due_slot": int(wo_row["promised_due_slot"]),
        "scheduling_priority_score": float(wo_row.get("scheduling_priority_score", wo_row["urgency_score"])),
        "on_schedule": on_schedule,
    }
    if on_schedule:
        s = sch_row.iloc[0]
        facts["assigned_technician"] = str(s["assigned_technician"])
        facts["start_slot"] = int(s["start_slot"])
        facts["end_slot"] = int(s["end_slot"])
        facts["lateness_slots"] = int(s.get("lateness_slots", 0))

    reasons: list[str] = []
    if float(wo_row["failure_risk"]) >= HIGH_RISK_MIN:
        reasons.append(f"high failure risk ({float(wo_row['failure_risk']):.2f})")
    if float(wo_row["urgency_score"]) >= URGENT_URGENCY_MIN:
        reasons.append(f"high urgency score ({float(wo_row['urgency_score']):.0f})")
    if _parts_ready(wo_row):
        reasons.append("required parts are available now (no parts ETA delay)")
    else:
        reasons.append(
            f"parts are not ready until slot {int(wo_row.get('parts_eta_slot', 0) or 0)}"
        )
    sec = str(wo_row.get("required_skill_secondary", "") or "").strip()
    if sec:
        reasons.append(f"needs dual skills ({wo_row['required_skill']} + {sec}) on a {wo_row['required_bay_type']} bay")
    else:
        reasons.append(f"skill/bay match: {wo_row['required_skill']} on {wo_row['required_bay_type']}")

    if on_schedule and int(sch_row.iloc[0]["start_slot"]) <= EARLY_SLOT_MAX:
        reasons.append("it is scheduled early in the planning window")

    if not reasons:
        reasons.append("standard priority relative to the current mix")

    if on_schedule:
        s = sch_row.iloc[0]
        answer = (
            f"{work_order_id} is prioritized because it has "
            + ", ".join(reasons)
            + f". It is assigned to {s['assigned_technician']} from slot {int(s['start_slot'])} to {int(s['end_slot'])}."
        )
    else:
        answer = (
            f"{work_order_id} is not on the current schedule output. "
            f"From work-order data: {', '.join(reasons)}."
        )

    return {
        "title": f"Why {work_order_id} is prioritized",
        "facts": facts,
        "answer": answer,
    }


def blocked_urgent_jobs(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    blocked = wo[
        (~wo.apply(_parts_ready, axis=1)) & (wo["urgency_score"].astype(float) >= URGENT_URGENCY_MIN)
    ].sort_values("urgency_score", ascending=False)

    if blocked.empty:
        return {
            "title": "Blocked urgent jobs",
            "facts": {"count": 0, "jobs": []},
            "answer": "No urgent jobs are currently blocked by parts (parts ETA > 0).",
        }

    sch_map = sched.set_index("work_order_id") if len(sched) else None
    jobs: list[dict[str, Any]] = []
    lines: list[str] = []
    for _, row in blocked.head(top_n).iterrows():
        wid = str(row["work_order_id"])
        eta = int(row.get("parts_eta_slot", 0) or 0)
        start_slot = None
        if sch_map is not None and wid in sch_map.index:
            start_slot = int(sch_map.loc[wid, "start_slot"])
        jobs.append(
            {
                "work_order_id": wid,
                "urgency_score": float(row["urgency_score"]),
                "vessel_id": str(row["vessel_id"]),
                "parts_eta_slot": eta,
                "start_slot": start_slot,
            }
        )
        if start_slot is not None:
            lines.append(
                f"{wid} (urgency {float(row['urgency_score']):.0f}, vessel {row['vessel_id']}, "
                f"parts ETA slot {eta}, work starts slot {start_slot})"
            )
        else:
            lines.append(
                f"{wid} (urgency {float(row['urgency_score']):.0f}, vessel {row['vessel_id']}, parts ETA slot {eta})"
            )

    count = int(len(blocked))
    answer = (
        f"{count} high-urgency job(s) are waiting on parts. "
        f"The most critical: " + "; ".join(lines) + "."
    )
    return {
        "title": "Blocked urgent jobs",
        "facts": {"count": count, "jobs": jobs},
        "answer": answer,
    }


def _tech_capacity_by_skill(techs: pd.DataFrame) -> dict[str, int]:
    out = {"engine": 0, "electrical": 0, "hull": 0, "general": 0}
    for _, t in techs.iterrows():
        if t.get("can_engine"):
            out["engine"] += 1
        if t.get("can_electrical"):
            out["electrical"] += 1
        if t.get("can_hull"):
            out["hull"] += 1
        if t.get("can_general"):
            out["general"] += 1
    return out


def summarize_bottlenecks(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    techs: pd.DataFrame,
    horizon_slots: float,
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "horizon_slots": float(horizon_slots),
        "n_technicians": len(techs),
        "n_scheduled_jobs": int(len(sched)),
    }

    urg = wo["urgency_score"].astype(float)
    high_urg = urg >= URGENT_URGENCY_MIN
    parts_wait = wo.apply(lambda r: not _parts_ready(r), axis=1)
    parts_blocked_urgent = int((high_urg & parts_wait).sum())
    facts["parts_blocked_urgent_count"] = parts_blocked_urgent

    cap = _tech_capacity_by_skill(techs)
    facts["technicians_by_skill"] = cap

    demand: dict[str, int] = {"engine": 0, "electrical": 0, "hull": 0, "general": 0}
    for _, r in wo.iterrows():
        sk = str(r["required_skill"])
        if sk in demand:
            if float(r["failure_risk"]) >= HIGH_RISK_MIN or float(r["urgency_score"]) >= URGENT_URGENCY_MIN:
                demand[sk] = demand.get(sk, 0) + 1
        sec = str(r.get("required_skill_secondary", "") or "").strip()
        if sec in demand:
            if float(r["failure_risk"]) >= HIGH_RISK_MIN or float(r["urgency_score"]) >= URGENT_URGENCY_MIN:
                demand[sec] = demand.get(sec, 0) + 1

    facts["hot_jobs_by_skill"] = demand
    ratios = {}
    for sk in demand:
        c = max(1, cap.get(sk, 0))
        ratios[sk] = round(demand[sk] / c, 2)
    facts["demand_to_capacity_ratio"] = ratios
    worst_skill = max(ratios, key=lambda k: ratios[k]) if ratios else "engine"

    # Utilization from schedule
    util_lines: list[str] = []
    if len(sched) and len(techs) and horizon_slots > 0:
        by_tech = sched.groupby("assigned_technician")["duration_slots"].sum()
        util = (by_tech / float(horizon_slots) * 100.0).sort_values(ascending=False)
        facts["top_utilization_pct"] = {str(k): float(round(v, 1)) for k, v in util.head(3).items()}
        top = util.index[0] if len(util) else None
        top_pct = float(util.iloc[0]) if len(util) else 0.0
        util_lines.append(
            f"Highest technician load is about {top_pct:.0f}% of horizon slots for {top}."
        )

    bay_counts = wo.groupby("required_bay_type").size().to_dict()
    facts["jobs_by_required_bay"] = {str(k): int(v) for k, v in bay_counts.items()}

    pieces = [
        f"The tightest skill capacity vs hot jobs appears to be **{worst_skill}** "
        f"({demand.get(worst_skill, 0)} critical/urgent jobs vs {cap.get(worst_skill, 0)} qualified techs on the roster)."
    ]
    if parts_blocked_urgent:
        pieces.append(
            f"{parts_blocked_urgent} urgent job(s) are still gated by parts ETAs."
        )
    if util_lines:
        pieces.append(util_lines[0])

    answer = " ".join(pieces)
    return {
        "title": "Today's biggest bottlenecks",
        "facts": facts,
        "answer": answer,
    }


def technician_utilization(
    sched: pd.DataFrame,
    techs: pd.DataFrame,
    horizon_slots: float,
) -> dict[str, Any]:
    if len(sched) == 0 or horizon_slots <= 0:
        return {
            "title": "Technician utilization",
            "facts": {},
            "answer": "No schedule rows to compute utilization.",
        }
    by_tech = sched.groupby("assigned_technician").agg(
        slots=("duration_slots", "sum"),
        jobs=("work_order_id", "count"),
    )
    by_tech["utilization_pct"] = (by_tech["slots"] / float(horizon_slots) * 100.0).round(1)
    by_tech = by_tech.sort_values("utilization_pct", ascending=False)
    top_name = str(by_tech.index[0])
    top_pct = float(by_tech["utilization_pct"].iloc[0])
    top_jobs = int(by_tech["jobs"].iloc[0])

    # Urgent engine concentration for top tech (from schedule merge would need wo — keep schedule-only)
    facts = {
        "horizon_slots": float(horizon_slots),
        "by_technician": [
            {
                "technician": str(idx),
                "utilization_pct": float(row["utilization_pct"]),
                "assigned_jobs": int(row["jobs"]),
                "duration_slots": int(row["slots"]),
            }
            for idx, row in by_tech.iterrows()
        ],
    }

    answer = (
        f"{top_name} is the most utilized at about {top_pct:.0f}% of horizon slots "
        f"({top_jobs} assignments). Rebalance or add capacity if that role is also carrying the urgent engine/electrical mix."
    )
    return {
        "title": "Technician utilization",
        "facts": facts,
        "answer": answer,
    }


def sla_risk_summary(
    wo: pd.DataFrame,
    sched: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    if len(sched) == 0:
        return {
            "title": "SLA / promised completion risk",
            "facts": {"at_risk": []},
            "answer": "No scheduled jobs — no SLA exposure to report.",
        }
    m = sched.merge(
        wo[
            [
                "work_order_id",
                "promised_due_slot",
                "urgency_score",
                "failure_risk",
            ]
        ],
        on="work_order_id",
        how="left",
    )
    late = m[m["lateness_slots"].astype(int) > 0].sort_values(
        "lateness_slots", ascending=False
    )
    if late.empty:
        return {
            "title": "SLA / promised completion risk",
            "facts": {"overdue_count": 0},
            "answer": "No jobs are projected past their promised completion slot in this plan.",
        }

    at_risk: list[dict[str, Any]] = []
    lines: list[str] = []
    for _, row in late.head(top_n).iterrows():
        late_s = int(row["lateness_slots"])
        at_risk.append(
            {
                "work_order_id": str(row["work_order_id"]),
                "end_slot": int(row["end_slot"]),
                "promised_due_slot": int(row["promised_due_slot"]),
                "lateness_slots": late_s,
            }
        )
        lines.append(
            f"{row['work_order_id']} finishes {late_s} slot(s) after promise (ends {int(row['end_slot'])}, due {int(row['promised_due_slot'])})"
        )

    worst = late.iloc[0]
    answer = (
        f"{len(late)} job(s) miss promised completion. "
        f"Most exposed: {worst['work_order_id']} ({int(worst['lateness_slots'])} slots late). "
        f"Details: " + "; ".join(lines) + "."
    )
    return {
        "title": "SLA / promised completion risk",
        "facts": {"overdue_count": int(len(late)), "at_risk": at_risk},
        "answer": answer,
    }


def executive_summary(
    kpis: dict[str, Any],
    sched: pd.DataFrame,
    wo: pd.DataFrame,
    horizon_slots: float,
    baseline_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_jobs = len(wo)
    span = float(sched["end_slot"].max()) if len(sched) else 0.0
    thr = np.percentile(wo[_priority_col(wo)].astype(float), 75) if len(wo) else 0.0
    high_pri = sched.merge(wo[["work_order_id", _priority_col(wo)]], on="work_order_id", how="left")
    if len(high_pri):
        hp = high_pri[high_pri[_priority_col(wo)].astype(float) >= thr]
        hp_done_early = float((hp["end_slot"] <= span / 3.0).mean() * 100.0) if len(hp) else 0.0
    else:
        hp_done_early = 0.0

    parts_pending = int((wo["parts_eta_slot"].astype(int) > 0).sum())
    urgent_parts = int(
        ((wo["urgency_score"].astype(float) >= URGENT_URGENCY_MIN) & (wo["parts_eta_slot"].astype(int) > 0)).sum()
    )

    facts = {
        "n_work_orders": n_jobs,
        "n_scheduled": int(len(sched)),
        "schedule_span_slots": span,
        "fleet_utilization_proxy": float(kpis.get("fleet_utilization_proxy", 0.0)),
        "total_sla_lateness_slots": float(kpis.get("total_sla_lateness_slots", 0.0)),
        "high_priority_on_time_pct": float(kpis.get("high_priority_on_time_pct", 0.0)),
        "high_priority_finished_in_first_third_pct": round(hp_done_early, 1),
        "work_orders_with_parts_eta": parts_pending,
        "urgent_jobs_waiting_parts": urgent_parts,
        "horizon_slots": float(horizon_slots),
    }
    if baseline_comparison:
        facts["baseline_snippet"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in baseline_comparison.items()
            if isinstance(v, (int, float, np.floating, str))
        }

    bline = ""
    if baseline_comparison and "overdue_count_reduction_pct" in baseline_comparison:
        bline = (
            f" Versus FCFS, overdue count is down about "
            f"{float(baseline_comparison['overdue_count_reduction_pct']):.0f}% in this demo metric."
        )

    answer = (
        f"The current schedule covers {int(len(sched))} work orders over about {span:.0f} slots "
        f"(horizon cap {horizon_slots:.0f}). "
        f"High-priority on-time rate is about {float(kpis.get('high_priority_on_time_pct', 0.0)):.0f}%; "
        f"total SLA lateness sums to {float(kpis.get('total_sla_lateness_slots', 0.0)):.0f} slots."
        f"{bline} "
        f"About {urgent_parts} urgent job(s) still wait on parts. "
        f"Fleet utilization proxy is {float(kpis.get('fleet_utilization_proxy', 0.0)) * 100:.0f}%."
    )
    return {
        "title": "Executive summary",
        "facts": facts,
        "answer": answer,
    }


def route_freeform_question(text: str) -> str | None:
    """Map a short user question to an action key; None = no match."""
    t = text.lower().strip()
    if not t:
        return None
    if any(x in t for x in ("blocked", "parts", "waiting on parts")):
        return "blocked"
    if any(x in t for x in ("bottleneck", "congestion", "capacity")):
        return "bottlenecks"
    if any(x in t for x in ("sla", "late", "overdue", "promised")):
        return "sla"
    if any(x in t for x in ("utilization", "overloaded", "busy tech", "technician")):
        return "utilization"
    if any(x in t for x in ("summarize", "summary", "executive", "overview")):
        return "executive"
    if any(x in t for x in ("why", "priorit", "explain", "first")):
        return "explain"
    return None


def get_gemini_api_key() -> str | None:
    _load_dotenv_from_project_root()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key.strip() if key and str(key).strip() else None


def polish_with_gemini(
    question: str,
    facts: dict[str, Any],
    draft_answer: str,
    api_key: str | None,
) -> str:
    """
    Rephrase the draft using only provided facts. Returns draft on failure / missing key.
    """
    if not api_key or not str(api_key).strip():
        return draft_answer
    try:
        import google.generativeai as genai

        _load_dotenv_from_project_root()
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
        payload = json.dumps(facts, indent=2, default=str)
        prompt = (
            "You are DockMaster AI Ops Assistant.\n"
            "Answer only using the facts provided below.\n"
            "Do not invent missing information.\n"
            "Be concise and operational (2–5 sentences).\n"
            "Preserve all numeric values exactly as in the facts or draft.\n\n"
            f"Question:\n{question}\n\n"
            f"Facts (JSON):\n{payload}\n\n"
            f"Draft answer (must remain factually equivalent):\n{draft_answer}"
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else draft_answer
    except Exception:
        return draft_answer
