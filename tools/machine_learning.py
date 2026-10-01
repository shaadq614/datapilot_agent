"""
ml_agent.py - A beginner-friendly, fully autonomous Machine Learning agent.

Usage:
    python ml_agent.py data.csv --target price
    python ml_agent.py data.xlsx --target churn --sheet Sheet1
    python ml_agent.py data.csv            # no --target => last column is used

What it does, with no human input:
    1. LOAD      read CSV / Excel safely
    2. SAFETY    check file size, shape, target quality
    3. PROFILE   describe every column
    4. PLAN      ask an LLM (optional) or use rules to choose a strategy
    5. VALIDATE  the plan is checked against an allow-list (the LLM cannot run code)
    6. TRAIN     compare several models against a dumb baseline
    7. EVALUATE  score the best one on data it has never seen
    8. REPORT    save model + metrics + plain-English report

The LLM is OPTIONAL. Without ANTHROPIC_API_KEY the agent uses built-in rules.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# SETTINGS - the safety limits. Change these numbers to be stricter or looser.
# ----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_MB = 50
MIN_ROWS = 30
MAX_ROWS = 1_000_000
MAX_COLUMNS = 500
MAX_TARGET_MISSING = 0.30   # refuse if more than 30% of target values are empty
MAX_COLUMN_MISSING = 0.60   # drop feature columns that are >60% empty
MAX_CATEGORIES = 50         # drop text columns with more unique values than this
LEAKAGE_DROP = 0.999        # drop features that are (almost) exact copies of the target
LEAKAGE_WARN = 0.95         # warn (but keep) features this strongly correlated
TIME_BUDGET_SECONDS = 300   # stop trying new models after 5 minutes
RANDOM_STATE = 42

# The ONLY models the agent is allowed to use (allow-list).
CLASSIFICATION_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
}
REGRESSION_MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}

# ----------------------------------------------------------------------------
# PROMPTS - what we tell the LLM. Edit these to change the agent's behaviour.
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the planning module of an automated machine learning agent.
You never write or run code. You only return a JSON plan.
You are careful, conservative and prefer simple, robust choices.
You only see column statistics, never raw rows."""

PLANNER_PROMPT = """Here is a profile of a dataset. The target column is "{target}".

PROFILE:
{profile}

ALLOWED MODELS:
- classification: {cls_models}
- regression: {reg_models}

Decide:
1. "task": "classification" or "regression" (based on the target's type and unique count).
2. "drop_columns": feature columns that should NOT be used because they look like
   identifiers, free text, names, dates stored as text, or could leak the answer
   (e.g. a column that is just another version of the target). Never include the target.
3. "models": 2 to 4 model names from the ALLOWED MODELS list for your chosen task.
4. "reasoning": one or two plain-English sentences.

Return ONLY valid JSON, no markdown, in exactly this shape:
{{"task": "...", "drop_columns": ["..."], "models": ["..."], "reasoning": "..."}}"""

EXPLAIN_PROMPT = """Explain these machine learning results to a non-technical reader in
5 short sentences or fewer. Say what was predicted, how good the result is compared with
the baseline, and one thing to be careful about. Do not invent numbers.

RESULTS:
{results}"""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
class SafetyError(Exception):
    """Raised when a safety check says the agent must stop."""


def log(step, message):
    print(f"[{step}] {message}")


def call_llm(system, user, max_tokens=800):
    """Ask the LLM something. Returns text, or None if unavailable (agent then uses rules)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # imported here so the agent works without it installed

        client = anthropic.Anthropic()
        reply = client.messages.create(
            model=os.getenv("AGENT_MODEL", "claude-sonnet-5-5"),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return reply.content[0].text
    except Exception as error:  # any failure => fall back to rules
        log("LLM", f"unavailable ({type(error).__name__}); using built-in rules instead")
        return None


def parse_json(text):
    """Pull a JSON object out of the LLM's reply, even if it added extra words."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------------
# STEP 1 + 2: LOAD with safety checks
# ----------------------------------------------------------------------------
def load_data(path, sheet=0):
    file = Path(path)
    if not file.exists():
        raise SafetyError(f"File not found: {file}")
    if file.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise SafetyError(f"Unsupported file type '{file.suffix}'. Use: {sorted(ALLOWED_EXTENSIONS)}")
    size_mb = file.stat().st_size / 1_000_000
    if size_mb > MAX_FILE_MB:
        raise SafetyError(f"File is {size_mb:.1f} MB; the limit is {MAX_FILE_MB} MB.")

    if file.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="latin-1")
    else:
        df = pd.read_excel(file, sheet_name=sheet)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all").drop_duplicates().reset_index(drop=True)

    if len(df) < MIN_ROWS:
        raise SafetyError(f"Only {len(df)} usable rows; need at least {MIN_ROWS}.")
    if len(df) > MAX_ROWS or df.shape[1] > MAX_COLUMNS:
        raise SafetyError(f"Data too large ({len(df)} rows x {df.shape[1]} columns).")
    if df.shape[1] < 2:
        raise SafetyError("Need at least 2 columns (features + target).")
    return df


def check_target(df, target):
    if target not in df.columns:
        raise SafetyError(f"Target '{target}' not found. Columns are: {list(df.columns)}")
    missing = df[target].isna().mean()
    if missing > MAX_TARGET_MISSING:
        raise SafetyError(f"Target is {missing:.0%} empty; the limit is {MAX_TARGET_MISSING:.0%}.")
    if df[target].nunique() < 2:
        raise SafetyError("Target has only one unique value; nothing to learn.")


# ----------------------------------------------------------------------------
# STEP 3: PROFILE (no raw rows are ever sent to the LLM)
# ----------------------------------------------------------------------------
def profile_data(df, target):
    columns = {}
    for name in df.columns:
        s = df[name]
        columns[name] = {
            "dtype": str(s.dtype),
            "missing_percent": round(float(s.isna().mean() * 100), 1),
            "unique_values": int(s.nunique()),
        }
    return {"rows": len(df), "target": target, "columns": columns}


# ----------------------------------------------------------------------------
# STEP 4 + 5: PLAN and VALIDATE
# ----------------------------------------------------------------------------
def infer_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def rule_based_plan(df, target):
    """The plan the agent makes by itself, without any LLM."""
    task = infer_task(df[target])
    drop = []
    for name in df.columns:
        if name == target:
            continue
        s = df[name]
        unique_ratio = s.nunique() / len(s)
        looks_like_id = name.lower() in {"id", "index"} or name.lower().endswith("_id")
        if s.isna().mean() > MAX_COLUMN_MISSING:
            drop.append(name)
        elif looks_like_id and unique_ratio > 0.9:
            drop.append(name)
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_CATEGORIES:
            drop.append(name)
    models = list(CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS)
    return {"task": task, "drop_columns": drop, "models": models, "reasoning": "Built-in rules."}


def llm_plan(profile, target):
    prompt = PLANNER_PROMPT.format(
        target=target,
        profile=json.dumps(profile, indent=1),
        cls_models=list(CLASSIFICATION_MODELS),
        reg_models=list(REGRESSION_MODELS),
    )
    return parse_json(call_llm(SYSTEM_PROMPT, prompt))


def validate_plan(plan, fallback, df, target):
    """SAFETY GATE: never trust the LLM. Keep only values that pass the allow-list."""
    task = plan.get("task")
    if task not in ("classification", "regression"):
        task = fallback["task"]
    # A numeric target with many values can't be classification; text target can't be regression.
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        task = "classification"

    allowed = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    models = [m for m in plan.get("models", []) if m in allowed]
    if not models:
        models = list(allowed)

    drops = [c for c in plan.get("drop_columns", []) if c in df.columns and c != target]
    # Always keep the rule-based drops too (union), but never drop everything.
    drops = sorted(set(drops) | set(fallback["drop_columns"]))
    if len(drops) >= df.shape[1] - 1:
        drops = fallback["drop_columns"]
    return {"task": task, "drop_columns": drops, "models": models,
            "reasoning": str(plan.get("reasoning", ""))[:300]}


def check_leakage(X, y, task):
    """SAFETY: find numeric features that (almost) equal the target.
    Returns (columns_to_drop, columns_to_warn_about)."""
    y_num = y if task == "regression" else pd.Series(pd.factorize(y)[0], index=y.index)
    drop, warn = [], []
    for name in X.select_dtypes(include="number").columns:
        corr = X[name].corr(y_num)
        if pd.isna(corr):
            continue
        if abs(corr) >= LEAKAGE_DROP:
            drop.append(name)
        elif abs(corr) >= LEAKAGE_WARN:
            warn.append(name)
    return drop, warn


# ----------------------------------------------------------------------------
# STEP 6: TRAIN
# ----------------------------------------------------------------------------
def build_pipeline(model, X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])


# ----------------------------------------------------------------------------
# THE AGENT
# ----------------------------------------------------------------------------
def run_agent(path, target=None, out_dir="agent_output", sheet=0):
    started = time.time()
    warnings_list = []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. Load + safety
    log("LOAD", f"reading {path}")
    df = load_data(path, sheet)
    target = target or df.columns[-1]
    log("LOAD", f"{len(df)} rows x {df.shape[1]} columns; target = '{target}'")
    check_target(df, target)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # 3. Profile
    profile = profile_data(df, target)

    # 4-5. Plan + validate
    fallback = rule_based_plan(df, target)
    raw_plan = llm_plan(profile, target)
    if raw_plan:
        plan = validate_plan(raw_plan, fallback, df, target)
        log("PLAN", f"LLM plan accepted after validation: {plan}")
    else:
        plan = fallback
        log("PLAN", f"rule-based plan: {plan}")
    task = plan["task"]

    X = df.drop(columns=[target] + plan["drop_columns"])
    y = df[target]
    if plan["drop_columns"]:
        warnings_list.append(f"Dropped columns (IDs/text/too empty): {plan['drop_columns']}")

    leaky, suspicious = check_leakage(X, y, task)
    if leaky:
        X = X.drop(columns=leaky)
        warnings_list.append(f"Dropped columns that are copies of the target: {leaky}")
    if suspicious:
        warnings_list.append(f"Kept but very strongly correlated with target (verify they are legitimate): {suspicious}")
    if X.shape[1] == 0:
        raise SafetyError("No usable feature columns remain after safety checks.")

    # Split data: the test set is locked away until the very end.
    stratify = None
    folds = 5
    if task == "classification":
        smallest_class = int(y.value_counts().min())
        if smallest_class < 2:
            raise SafetyError("A target class has only 1 row; cannot split data fairly.")
        stratify = y
        folds = max(2, min(5, smallest_class))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    if task == "classification":
        folds = max(2, min(folds, int(y_train.value_counts().min())))

    # 6. Train + compare (baseline first)
    scoring = "f1_macro" if task == "classification" else "r2"
    catalog = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    baseline = DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()

    results = {}
    candidates = [("baseline", lambda: baseline)] + [(m, catalog[m]) for m in plan["models"]]
    for name, factory in candidates:
        if time.time() - started > TIME_BUDGET_SECONDS:
            warnings_list.append(f"Time budget reached; skipped remaining models starting at '{name}'.")
            break
        try:
            scores = cross_val_score(build_pipeline(factory(), X_train), X_train, y_train,
                                     cv=folds, scoring=scoring)
            results[name] = float(scores.mean())
            log("TRAIN", f"{name:<20} {scoring} = {results[name]:.3f}")
        except Exception as error:
            warnings_list.append(f"Model '{name}' failed and was skipped: {type(error).__name__}")

    real = {k: v for k, v in results.items() if k != "baseline"}
    if not real:
        raise SafetyError("No model trained successfully.")
    best_name = max(real, key=real.get)
    if real[best_name] <= results.get("baseline", -np.inf) + 0.01:
        warnings_list.append("Best model barely beats the baseline: the data may not contain a useful signal.")

    # 7. Final evaluation on untouched test data
    final = build_pipeline(catalog[best_name](), X_train).fit(X_train, y_train)
    pred = final.predict(X_test)
    if task == "classification":
        metrics = {"accuracy": accuracy_score(y_test, pred),
                   "f1_macro": f1_score(y_test, pred, average="macro")}
        headline = metrics["f1_macro"]
    else:
        metrics = {"r2": r2_score(y_test, pred),
                   "mae": mean_absolute_error(y_test, pred),
                   "rmse": float(np.sqrt(mean_squared_error(y_test, pred)))}
        headline = metrics["r2"]
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    if headline > 0.99:
        warnings_list.append("Score above 0.99 is suspicious: check for data leakage.")
    log("EVALUATE", f"best = {best_name}; test metrics = {metrics}")

    # 8. Report + save (only inside the output folder; input file is never modified)
    joblib.dump(final, out / "model.joblib")
    summary = {
        "task": task, "target": target, "best_model": best_name,
        "features_used": list(X.columns), "cv_scores": {k: round(v, 4) for k, v in results.items()},
        "test_metrics": metrics, "warnings": warnings_list,
        "plan_reasoning": plan["reasoning"], "seconds": round(time.time() - started, 1),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    explanation = call_llm(SYSTEM_PROMPT, EXPLAIN_PROMPT.format(results=json.dumps(summary, indent=1)), 400)
    lines = [f"# ML Agent Report", "",
             f"- **Task:** {task}", f"- **Target:** {target}", f"- **Best model:** {best_name}",
             f"- **Test metrics:** {metrics}", f"- **Baseline CV score:** {results.get('baseline'):.3f}", ""]
    if explanation:
        lines += ["## Plain-English summary", explanation, ""]
    if warnings_list:
        lines += ["## Warnings"] + [f"- {w}" for w in warnings_list] + [""]
    lines += ["## Model comparison (cross-validation)"] + [f"- {k}: {v:.3f}" for k, v in results.items()]
    (out / "report.md").write_text("\n".join(lines))
    log("REPORT", f"saved model.joblib, metrics.json, report.md in '{out}/'")
    for w in warnings_list:
        log("WARNING", w)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML agent for CSV/Excel files.")
    parser.add_argument("file", help="Path to a .csv, .xlsx or .xls file")
    parser.add_argument("--target", help="Column to predict (default: last column)")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first)")
    parser.add_argument("--out", default="agent_output", help="Output folder")
    args = parser.parse_args()
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    try:
        run_agent(args.file, args.target, args.out, sheet)
    except SafetyError as error:
        print(f"[STOPPED BY SAFETY CHECK] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
