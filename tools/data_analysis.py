"""
Data Analysis Agent (beginner-friendly, single file)

Setup:
    pip install anthropic pandas numpy matplotlib openpyxl
    export ANTHROPIC_API_KEY="your-key"        (Windows: set ANTHROPIC_API_KEY=your-key)

Run:
    python agent.py mydata.csv "Which region has the best sales growth?"
    python agent.py mydata.xlsx            (no question = general exploratory analysis)

Outputs go to ./output/<timestamp>/ : report.md, charts (*.png), log.jsonl
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd

# ----------------------------------------------------------------------------
# 1. SETTINGS (your safety limits live here)
# ----------------------------------------------------------------------------
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 12               # hard cap on agent loop iterations
CODE_TIMEOUT_SEC = 30        # kill any code that runs longer than this
MAX_FILE_MB = 50             # refuse huge files
MAX_OUTPUT_CHARS = 4000      # truncate tool output sent back to the model

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "scipy",
    "math", "statistics", "datetime", "re", "json", "collections",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}
BANNED_ATTRS = {
    # file / network / system access through pandas or other libs
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql",
    "read_parquet", "read_html", "read_clipboard",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "to_html", "to_clipboard", "to_feather", "to_hdf",
    "system", "popen", "remove", "unlink", "rmdir", "rmtree", "rename",
    "urlopen", "urlretrieve",
}

# ----------------------------------------------------------------------------
# 2. PROMPTS
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful, autonomous data analyst.

GOAL: Answer the user's question (or explore the data if no question is given)
using the tools provided, then deliver a clear final report.

WORKFLOW
1. PLAN: State a short plan (3-5 steps) before running code.
2. INSPECT: Check data quality first (missing values, duplicates, odd types, outliers).
3. ANALYZE: Run small, focused pieces of code. One idea per run_python call.
4. VERIFY: Sanity-check surprising results (row counts, totals, units) before trusting them.
5. REPORT: Call finish_report with your findings.

CODE RULES (for run_python)
- Pre-loaded: `df` (first sheet / the CSV), `sheets` (dict of all sheets), pd, np, plt.
- Each call is a fresh process: variables do NOT persist. Write self-contained code.
- Use print() to see results. Keep printed output short (use .head(), .describe()).
- Save charts with plt.savefig("descriptive_name.png", dpi=120, bbox_inches="tight"), then plt.close().
- Only these imports are allowed: pandas, numpy, matplotlib, seaborn, scipy, math, statistics, datetime, re, json, collections.
- You cannot read/write other files, use the network, or modify the original data.
- If code is rejected or errors, read the message, fix it, and retry (max 2 retries per problem).

HONESTY RULES
- Never invent numbers. Every figure in the report must come from code output.
- State assumptions and data-quality issues plainly.
- Correlation is not causation; say so where relevant.
- If the data cannot answer the question, say that instead of guessing.

FINAL REPORT FORMAT (markdown)
# Title
## Answer (2-3 sentences)
## Key findings (bullets with numbers)
## Data quality notes
## Charts (list filenames)
## Assumptions & limitations
## Suggested next steps
"""

TOOLS = [
    {
        "name": "run_python",
        "description": "Run Python code against the loaded data. Returns printed output or an error.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Self-contained Python code."}},
            "required": ["code"],
        },
    },
    {
        "name": "finish_report",
        "description": "Submit the final markdown report. Ends the session.",
        "input_schema": {
            "type": "object",
            "properties": {"report_markdown": {"type": "string"}},
            "required": ["report_markdown"],
        },
    },
]


# ----------------------------------------------------------------------------
# 3. SAFETY CHECKS
# ----------------------------------------------------------------------------
def check_file(path: Path) -> None:
    """Validate the input file before anything else happens."""
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        sys.exit("Only .csv, .xlsx and .xls files are supported.")
    size_mb = path.stat().st_size / 1_000_000
    if size_mb > MAX_FILE_MB:
        sys.exit(f"File is {size_mb:.1f} MB; limit is {MAX_FILE_MB} MB.")


def check_code(code: str) -> str | None:
    """Static check of AI-written code. Returns an error message, or None if OK."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"Import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"Import not allowed: {node.module}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"Blocked function/name: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                return f"Blocked attribute: .{node.attr}"
            if node.attr.startswith("__"):
                return f"Blocked dunder attribute: .{node.attr}"
    return None


# ----------------------------------------------------------------------------
# 4. CODE EXECUTION (separate process + timeout + own output folder)
# ----------------------------------------------------------------------------
def make_preamble(data_path: Path) -> str:
    """Code we control: loads the data fresh each run so the original is never changed."""
    return f'''
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_p = r"{data_path.resolve()}"
if _p.lower().endswith(".csv"):
    sheets = {{"data": pd.read_csv(_p)}}
else:
    sheets = pd.read_excel(_p, sheet_name=None)
df = next(iter(sheets.values()))
'''


def run_python(code: str, data_path: Path, out_dir: Path) -> str:
    problem = check_code(code)
    if problem:
        return f"REJECTED by safety check: {problem}. Rewrite the code without it."

    full_code = make_preamble(data_path) + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=out_dir) as f:
        f.write(full_code)
        script = f.name
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=out_dir,                      # charts land here
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT_SEC,
        )
        output = result.stdout + (("\nERROR:\n" + result.stderr[-1500:]) if result.returncode else "")
    except subprocess.TimeoutExpired:
        output = f"TIMEOUT: code ran longer than {CODE_TIMEOUT_SEC}s. Use a simpler approach."
    finally:
        os.unlink(script)

    output = output.strip() or "(no output - did you forget print()?)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    return output


# ----------------------------------------------------------------------------
# 5. DATA PROFILE (gives the model a head start)
# ----------------------------------------------------------------------------
def quick_profile(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        sheets = {"data": pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for name, d in sheets.items():
        parts.append(
            f"Sheet '{name}': {d.shape[0]} rows x {d.shape[1]} cols\n"
            f"Columns/types:\n{d.dtypes.to_string()}\n"
            f"Missing values:\n{d.isna().sum()[d.isna().sum() > 0].to_string() or 'none'}\n"
            f"First 5 rows:\n{d.head().to_string()}"
        )
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# 6. THE AGENT LOOP
# ----------------------------------------------------------------------------
def log(out_dir: Path, event: dict) -> None:
    with open(out_dir / "log.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def run(data_path: Path, question: str, out_dir: Path) -> Path | None:
    """Run the analysis agent. Returns the report path, or None if no report was written.
    (Called by main.py; the command-line entry point is main() below.)"""
    data_path, out_dir = Path(data_path), Path(out_dir)
    check_file(data_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    messages = [{
        "role": "user",
        "content": f"QUESTION: {question}\n\nDATA PROFILE:\n{quick_profile(data_path)}",
    }]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step}/{MAX_STEPS} ---")
        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(block.text)
            elif block.type == "tool_use":
                if block.name == "run_python":
                    code = block.input["code"]
                    print(f"[running code]\n{code}\n")
                    result = run_python(code, data_path, out_dir)
                    print(f"[result]\n{result}")
                    log(out_dir, {"step": step, "code": code, "result": result})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                elif block.name == "finish_report":
                    (out_dir / "report.md").write_text(block.input["report_markdown"])
                    print(f"\nDone. Report saved to {out_dir / 'report.md'}")
                    return out_dir / "report.md"

        if not tool_results:   # model replied with text only and no tool call
            messages.append({"role": "user", "content": "Continue, or call finish_report if you are done."})
        else:
            messages.append({"role": "user", "content": tool_results})

    # Safety net: step limit reached
    print("\nStep limit reached. Asking for a final report with what was found so far...")
    messages.append({"role": "user", "content": "Step limit reached. Call finish_report now with your findings so far, noting what is incomplete."})
    response = client.messages.create(
        model=MODEL, max_tokens=3000, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        tool_choice={"type": "tool", "name": "finish_report"},
    )
    for block in response.content:
        if block.type == "tool_use":
            (out_dir / "report.md").write_text(block.input["report_markdown"])
            print(f"Partial report saved to {out_dir / 'report.md'}")
            return out_dir / "report.md"
    return None


def main() -> None:
    """Standalone command-line entry point (same behaviour as before)."""
    if len(sys.argv) < 2:
        sys.exit('Usage: python data_analysis.py <file.csv|xlsx> ["your question"]')
    data_path = Path(sys.argv[1])
    question = sys.argv[2] if len(sys.argv) > 2 else "Do a general exploratory analysis and highlight the most important insights."
    out_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run(data_path, question, out_dir)


if __name__ == "__main__":
    main()