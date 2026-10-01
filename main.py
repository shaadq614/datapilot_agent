"""
main.py - Orchestrator for the autonomous data science agent.

Pipeline:  VALIDATE -> CLEAN -> (resolve target) -> ANALYZE -> VISUALIZE -> MODEL -> REPORT

Each stage is one of your tool agents in tools/. The cleaned file produced by the
cleaning stage is the input to every later stage. A failing stage is logged and
skipped so the rest of the pipeline still runs (except loading/target checks).

Usage:
    python main.py datasets/sales.csv
    python main.py datasets/sales.csv --target revenue
    python main.py datasets/sales.csv --target revenue --question "What drives revenue?"
    python main.py datasets/sales.csv --skip analysis visualization     # no LLM-heavy stages

Notes:
    * Cleaning and ML work without an API key (rule-based fallback).
    * Analysis and Visualization are LLM agents and need ANTHROPIC_API_KEY;
      without it they are skipped automatically.
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Paths (relative to this file, so it works from any working directory)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_DIR = OUTPUT_DIR / "cleaned_data"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIZ_DIR = OUTPUT_DIR / "visualization"
MODEL_DIR = OUTPUT_DIR / "model"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}   # the cleaning agent supports these two
STAGES = ("clean", "analysis", "visualization", "ml")

for _d in (CLEANED_DIR, ANALYSIS_DIR, VIZ_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ds_agent")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_tool(name: str):
    """Import a tools/ module only when needed, so a missing library
    (e.g. anthropic) only breaks the stage that uses it, not the whole app."""
    return importlib.import_module(f"tools.{name}")


def normalise_name(col) -> str:
    """Same renaming rule the cleaning agent applies to column names."""
    return str(col).strip().lower().replace(" ", "_")


def read_columns(path: Path) -> list:
    """Read only the header of a csv/xlsx file (first sheet)."""
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, nrows=0)
        except UnicodeDecodeError:
            df = pd.read_csv(path, nrows=0, encoding="latin-1")
    else:
        df = pd.read_excel(path, nrows=0)
    return [str(c).strip() for c in df.columns]


def resolve_target(requested, columns: list) -> str:
    """Match the user's --target against the (possibly renamed) cleaned columns."""
    if requested is None:
        return columns[-1]                      # same default the ML agent uses
    if requested in columns:
        return requested
    if normalise_name(requested) in columns:    # cleaning lowercases + underscores names
        return normalise_name(requested)
    by_lower = {c.lower(): c for c in columns}
    if requested.lower() in by_lower:
        return by_lower[requested.lower()]
    raise ValueError(f"Target '{requested}' not found. Available columns: {columns}")


def run_stage(name: str, func, report: dict, critical: bool = False):
    """Run one stage with timing + error handling. Non-critical failures are recorded and skipped."""
    log.info("=== %s ===", name.upper())
    start = time.time()
    try:
        result = func()
        report["stages"][name] = {"status": "success", "seconds": round(time.time() - start, 1)}
        return result
    except (Exception, SystemExit) as exc:      # tools call sys.exit() on bad input
        message = str(exc) or type(exc).__name__
        log.error("Stage '%s' failed: %s", name, message)
        log.debug(traceback.format_exc())
        report["stages"][name] = {"status": "failed", "error": message,
                                  "seconds": round(time.time() - start, 1)}
        if critical:
            raise
        return None


def mark_skipped(report: dict, name: str, reason: str):
    log.warning("Skipping %s: %s", name, reason)
    report["stages"][name] = {"status": "skipped", "reason": reason, "seconds": 0}


# ----------------------------------------------------------------------------
# Stage implementations (thin wrappers around your tools/*.py)
# ----------------------------------------------------------------------------
def stage_clean(data_path: Path, use_llm: bool) -> dict:
    dc = load_tool("data_cleaning")
    dc.OUTPUT_DIR = CLEANED_DIR                 # tool saves here instead of ./output

    tables = dc.load_file(data_path)
    cleaned, logs, rows_before = {}, [], 0
    for name, df in tables.items():
        rows_before += len(df)
        cleaned[name], table_log = dc.clean_table(name, df, use_llm)
        logs.append(table_log)

    out_file, log_file = dc.save_outputs(data_path, cleaned, logs)
    counts = {}
    for t in logs:
        for step in t["steps"]:
            counts[step["status"]] = counts.get(step["status"], 0) + 1
    return {
        "cleaned_file": str(out_file),
        "log_file": str(log_file),
        "rows_before": rows_before,
        "rows_after": sum(len(d) for d in cleaned.values()),
        "step_outcomes": counts,
        "mode": "AI-planned" if use_llm else "rule-based",
    }


def stage_analysis(data_path: Path, question: str, run_id: str):
    da = load_tool("data_analysis")
    return da.run(data_path, question, ANALYSIS_DIR / run_id)


def stage_visualization(data_path: Path, request: str, run_id: str):
    dv = load_tool("data_visualization")
    return dv.run(data_path, request, VIZ_DIR / run_id)


def stage_ml(data_path: Path, target: str, run_id: str) -> dict:
    ml = load_tool("machine_learning")
    return ml.run_agent(str(data_path), target=target, out_dir=str(MODEL_DIR / run_id))


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def write_reports(report: dict) -> Path:
    run_id = report["run_id"]
    json_path = REPORT_DIR / f"run_{run_id}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [f"# Run {run_id}", "",
             f"- **Dataset:** `{report['dataset']}`",
             f"- **Target:** `{report.get('target')}`",
             f"- **Question:** {report.get('question')}", "",
             "## Stages", "", "| Stage | Status | Seconds | Notes |", "|---|---|---|---|"]
    for name, info in report["stages"].items():
        note = info.get("error") or info.get("reason") or ""
        lines.append(f"| {name} | {info['status']} | {info.get('seconds', 0)} | {note} |")

    cleaning = report.get("cleaning")
    if cleaning:
        lines += ["", "## Cleaning", f"- Rows: {cleaning['rows_before']} -> {cleaning['rows_after']}",
                  f"- Step outcomes: {cleaning['step_outcomes']}",
                  f"- Cleaned file: `{cleaning['cleaned_file']}`"]
    model = report.get("model")
    if model:
        lines += ["", "## Model", f"- Task: {model['task']}", f"- Best model: {model['best_model']}",
                  f"- Test metrics: {model['test_metrics']}",
                  f"- CV scores: {model['cv_scores']}"]
        for w in model.get("warnings", []):
            lines.append(f"- WARNING: {w}")
    outputs = report.get("outputs", {})
    if outputs:
        lines += ["", "## Output files"] + [f"- {k}: `{v}`" for k, v in outputs.items()]

    (REPORT_DIR / f"run_{run_id}.md").write_text("\n".join(lines), encoding="utf-8")
    return json_path


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(data_path: Path, target, question, skip: set) -> dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    report = {"run_id": run_id, "dataset": str(data_path), "stages": {}, "outputs": {}}
    work_path = data_path                       # file fed to every stage after cleaning

    # 1. Clean -----------------------------------------------------------
    if "clean" in skip:
        mark_skipped(report, "cleaning", "skipped by user")
    else:
        result = run_stage("cleaning", lambda: stage_clean(data_path, has_key), report)
        if result:
            report["cleaning"] = result
            work_path = Path(result["cleaned_file"])
            report["outputs"]["cleaned_data"] = result["cleaned_file"]
        else:
            log.warning("Cleaning failed - continuing with the raw file.")

    # 2. Resolve target (names may have changed during cleaning) -----------
    def _resolve():
        return resolve_target(target, read_columns(work_path))
    resolved_target = run_stage("target_check", _resolve, report, critical=True)
    report["target"] = resolved_target
    log.info("Target column: '%s'", resolved_target)

    if question:
        analysis_q = viz_q = question
    else:
        analysis_q = (f"Do a general exploratory analysis. The dataset will be used to predict "
                      f"'{resolved_target}', so highlight what relates to it.")
        viz_q = (f"Choose the most insightful charts. Include charts showing how the main "
                 f"features relate to '{resolved_target}'.")
    report["question"] = question or "(default exploratory analysis)"

    # 3. Analysis (LLM agent) ---------------------------------------------
    if "analysis" in skip:
        mark_skipped(report, "analysis", "skipped by user")
    elif not has_key:
        mark_skipped(report, "analysis", "ANTHROPIC_API_KEY not set")
    else:
        path = run_stage("analysis", lambda: stage_analysis(work_path, analysis_q, run_id), report)
        if path:
            report["outputs"]["analysis_report"] = str(path)

    # 4. Visualization (LLM agent) -----------------------------------------
    if "visualization" in skip:
        mark_skipped(report, "visualization", "skipped by user")
    elif not has_key:
        mark_skipped(report, "visualization", "ANTHROPIC_API_KEY not set")
    else:
        path = run_stage("visualization", lambda: stage_visualization(work_path, viz_q, run_id), report)
        if path:
            report["outputs"]["visualization_gallery"] = str(path)

    # 5. Machine learning ----------------------------------------------------
    if "ml" in skip:
        mark_skipped(report, "ml", "skipped by user")
    else:
        summary = run_stage("ml", lambda: stage_ml(work_path, resolved_target, run_id), report)
        if summary:
            report["model"] = summary
            report["outputs"]["model_dir"] = str(MODEL_DIR / run_id)

    report_path = write_reports(report)
    report["outputs"]["run_report"] = str(report_path)
    log.info("Run report: %s", report_path)
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Data Science Agent")
    p.add_argument("data", help="Path to a .csv or .xlsx file (or just a filename inside datasets/)")
    p.add_argument("--target", default=None, help="Column to predict (default: last column)")
    p.add_argument("--question", default=None, help="Question for the analysis/visualization agents")
    p.add_argument("--skip", nargs="*", choices=STAGES, default=[],
                   help="Stages to skip, e.g. --skip analysis visualization")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists() and not data_path.is_absolute():
        data_path = DATASET_DIR / args.data
    if not data_path.exists():
        sys.exit(f"Dataset not found: {args.data}")
    if data_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        sys.exit(f"Unsupported file type '{data_path.suffix}'. Use: {sorted(SUPPORTED_EXTENSIONS)}")

    try:
        report = run_pipeline(data_path.resolve(), args.target, args.question, set(args.skip))
    except ValueError as exc:                   # e.g. target column not found
        sys.exit(f"Stopped: {exc}")

    print("\n" + "=" * 56 + "\nPIPELINE SUMMARY\n" + "=" * 56)
    for name, info in report["stages"].items():
        extra = info.get("error") or info.get("reason") or ""
        print(f"{name:<14} {info['status']:<8} {info.get('seconds', 0):>6}s  {extra}")
    if report.get("model"):
        m = report["model"]
        print(f"\nBest model: {m['best_model']} ({m['task']})  test metrics: {m['test_metrics']}")
    print("\nOutputs:")
    for k, v in report["outputs"].items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()