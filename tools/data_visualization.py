"""
Data Visualization Agent (beginner-friendly, single file)

Setup:
    pip install anthropic pandas numpy matplotlib seaborn openpyxl
    export ANTHROPIC_API_KEY="your-key"        (Windows: set ANTHROPIC_API_KEY=your-key)

Run:
    python viz_agent.py sales.csv "Show revenue trends and top products"
    python viz_agent.py sales.xlsx             (no request = pick the most useful charts)

Outputs go to ./output_viz/<timestamp>/ : charts (*.png), index.html gallery, log.jsonl

KEY IDEA: the agent LOOKS at each chart it makes (image fed back to the model),
critiques it, and redraws if it's ugly or misleading.
"""

import ast
import base64
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 16               # hard cap on loop iterations
MAX_CHARTS = 6               # hard cap on kept charts
CODE_TIMEOUT_SEC = 30
MAX_FILE_MB = 50
MAX_OUTPUT_CHARS = 3000
MAX_IMAGE_MB = 1.5           # don't send giant images back to the model

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "scipy",
    "math", "statistics", "datetime", "re", "json", "collections",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}
BANNED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql",
    "read_parquet", "read_html", "read_clipboard",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "to_html", "to_clipboard", "to_feather", "to_hdf",
    "system", "popen", "remove", "unlink", "rmdir", "rmtree", "rename",
    "urlopen", "urlretrieve",
    "savefig",   # the agent never saves files itself; our wrapper does it safely
    "show",      # no windows
}

# ----------------------------------------------------------------------------
# 2. PROMPTS
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert data visualization designer working autonomously.

GOAL: Turn the user's data into a small set of clear, honest, well-designed charts
that answer their request (or reveal the most important patterns if no request is given).

WORKFLOW
1. PLAN: Briefly list the charts you intend to make and WHY each one (what question it answers).
2. INSPECT: Use run_python to check column types, ranges, missing values, cardinality.
3. CREATE: Use create_chart for each chart. You will be SHOWN the rendered image.
4. CRITIQUE: Look at every image. Check the checklist below. If anything fails, call
   create_chart again with the SAME filename to replace it (max 2 revisions per chart).
5. FINISH: When done, call finish with a short summary of what each chart shows.

CHART SELECTION GUIDE
- Change over time -> line chart (sort by date first)
- Compare categories -> horizontal bar chart, sorted by value
- Distribution of one number -> histogram or box plot
- Relationship between two numbers -> scatter plot (use alpha for many points)
- Part-of-whole -> bar chart; pie/donut ONLY if 5 or fewer slices
- Many categories -> show top N (e.g. 10) and group the rest as "Other"
- Two numeric-vs-category dimensions -> grouped bars or heatmap
Never use 3D charts, dual y-axes, or decorative effects.

DESIGN CHECKLIST (verify by looking at the image)
[ ] Descriptive title that states the takeaway (e.g. "Q4 revenue grew 23% over Q3")
[ ] Axis labels with units; readable tick labels (no overlap; rotate or go horizontal)
[ ] Bar charts start at zero
[ ] Legend only when needed and not covering data
[ ] Colorblind-safe palette, max ~6 colors
[ ] No clutter: light gridlines, no chart junk
[ ] Numbers in the title/annotations match the data (compute them in code, never guess)

CODE RULES
- Pre-loaded: `df` (first sheet / CSV), `sheets` (dict of all sheets), pd, np, plt, sns.
- Each call is a fresh process: variables do NOT persist. Write self-contained code.
- For create_chart: build the figure with plt (use fig, ax = plt.subplots(figsize=(9, 5.5))).
  DO NOT call savefig or show. The system saves the figure for you.
- Clean data inside your code when needed (parse dates, drop NaN for that chart) and say so in notes.
- Allowed imports: pandas, numpy, matplotlib, seaborn, scipy, math, statistics, datetime, re, json, collections.
- No file access, no network. If rejected, fix and retry.

HONESTY RULES
- Do not cherry-pick axes or truncate data to exaggerate a trend.
- If you filter, aggregate, or drop rows, mention it in the chart's `notes`.
- If the data can't support the requested chart, say so instead of forcing it.
"""

TOOLS = [
    {
        "name": "run_python",
        "description": "Inspect the data with Python (print results). Not for making charts.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Build a matplotlib/seaborn chart. Code must create the figure but NOT save/show it. "
            "Returns the rendered image so you can review it. Reusing a filename replaces that chart."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "lowercase letters, digits, underscores only, no extension"},
                "code": {"type": "string"},
                "purpose": {"type": "string", "description": "One sentence: what question this chart answers"},
                "notes": {"type": "string", "description": "Any filtering/cleaning done for this chart"},
            },
            "required": ["filename", "code", "purpose"],
        },
    },
    {
        "name": "finish",
        "description": "Submit the final summary. Ends the session.",
        "input_schema": {
            "type": "object",
            "properties": {"summary_markdown": {"type": "string"}},
            "required": ["summary_markdown"],
        },
    },
]


# ----------------------------------------------------------------------------
# 3. SAFETY CHECKS
# ----------------------------------------------------------------------------
def check_file(path: Path) -> None:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        sys.exit("Only .csv, .xlsx and .xls files are supported.")
    size_mb = path.stat().st_size / 1_000_000
    if size_mb > MAX_FILE_MB:
        sys.exit(f"File is {size_mb:.1f} MB; limit is {MAX_FILE_MB} MB.")


def check_code(code: str) -> str | None:
    """Static check of AI-written code. Returns error message, or None if OK."""
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
            return f"Blocked name: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                return f"Blocked attribute: .{node.attr}"
            if node.attr.startswith("__"):
                return f"Blocked dunder attribute: .{node.attr}"
    return None


def safe_filename(name: str) -> str | None:
    """Only simple names allowed: prevents writing outside the output folder."""
    name = name.strip().lower().removesuffix(".png")
    return name if re.fullmatch(r"[a-z0-9_]{1,50}", name) else None


# ----------------------------------------------------------------------------
# 4. EXECUTION (separate process, timeout, our own output folder)
# ----------------------------------------------------------------------------
def make_preamble(data_path: Path) -> str:
    return f'''
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="colorblind")
_p = r"{data_path.resolve()}"
if _p.lower().endswith(".csv"):
    sheets = {{"data": pd.read_csv(_p)}}
else:
    sheets = pd.read_excel(_p, sheet_name=None)
df = next(iter(sheets.values()))
'''


def _run(script_text: str, out_dir: Path) -> tuple[bool, str]:
    """Run a script in a subprocess. Returns (success, output)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=out_dir) as f:
        f.write(script_text)
        script = f.name
    try:
        r = subprocess.run([sys.executable, script], cwd=out_dir,
                           capture_output=True, text=True, timeout=CODE_TIMEOUT_SEC)
        ok = r.returncode == 0
        out = r.stdout + ("" if ok else "\nERROR:\n" + r.stderr[-1500:])
    except subprocess.TimeoutExpired:
        ok, out = False, f"TIMEOUT: code ran longer than {CODE_TIMEOUT_SEC}s."
    finally:
        os.unlink(script)
    out = out.strip() or "(no output)"
    return ok, out[:MAX_OUTPUT_CHARS]


def run_python(code: str, data_path: Path, out_dir: Path) -> str:
    problem = check_code(code)
    if problem:
        return f"REJECTED by safety check: {problem}."
    return _run(make_preamble(data_path) + "\n" + code, out_dir)[1]


def create_chart(inp: dict, data_path: Path, out_dir: Path, charts: dict):
    """Returns tool_result content (list with image + text, or plain text on error)."""
    name = safe_filename(inp["filename"])
    if not name:
        return "REJECTED: filename must be lowercase letters/digits/underscores only."
    if name not in charts and len(charts) >= MAX_CHARTS:
        return f"REJECTED: chart limit ({MAX_CHARTS}) reached. Replace an existing chart or call finish."
    problem = check_code(inp["code"])
    if problem:
        return f"REJECTED by safety check: {problem}."

    png_path = out_dir / f"{name}.png"
    # We add the save step ourselves, so the model never touches the filesystem.
    script = (
        make_preamble(data_path) + "\n" + inp["code"] +
        f'\nplt.tight_layout()\nplt.savefig(r"{png_path}", dpi=110, bbox_inches="tight")\nplt.close("all")\n'
    )
    ok, out = _run(script, out_dir)
    if not ok or not png_path.exists():
        return f"Chart failed:\n{out}"

    size_mb = png_path.stat().st_size / 1_000_000
    charts[name] = {"purpose": inp["purpose"], "notes": inp.get("notes", "")}
    if size_mb > MAX_IMAGE_MB:
        return f"Chart saved as {name}.png but it is too large to preview ({size_mb:.1f} MB). Simplify it."

    img_b64 = base64.standard_b64encode(png_path.read_bytes()).decode()
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
        {"type": "text", "text": f"Rendered {name}.png. Review it against the design checklist. "
                                 f"Revise (same filename) if needed, otherwise continue."},
    ]


# ----------------------------------------------------------------------------
# 5. HELPERS
# ----------------------------------------------------------------------------
def quick_profile(path: Path) -> str:
    sheets = {"data": pd.read_csv(path)} if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=None)
    parts = []
    for name, d in sheets.items():
        missing = d.isna().sum()
        parts.append(
            f"Sheet '{name}': {d.shape[0]} rows x {d.shape[1]} cols\n"
            f"Types:\n{d.dtypes.to_string()}\n"
            f"Missing:\n{missing[missing > 0].to_string() or 'none'}\n"
            f"Head:\n{d.head().to_string()}"
        )
    return "\n\n".join(parts)


def log(out_dir: Path, event: dict) -> None:
    with open(out_dir / "log.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def write_gallery(out_dir: Path, charts: dict, summary: str) -> None:
    """Simple HTML page you can open in any browser."""
    cards = "".join(
        f'<figure><img src="{n}.png" style="max-width:100%">'
        f'<figcaption><b>{html.escape(n.replace("_", " ").title())}</b><br>'
        f'{html.escape(c["purpose"])}'
        f'{"<br><i>Notes: " + html.escape(c["notes"]) + "</i>" if c["notes"] else ""}'
        f'</figcaption></figure>'
        for n, c in charts.items()
    )
    page = (
        '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Charts</title></head>'
        '<body style="font-family:sans-serif;max-width:900px;margin:auto;padding:16px">'
        f'<h1>Data Visualization Report</h1>'
        f'<pre style="white-space:pre-wrap;font-family:inherit">{html.escape(summary)}</pre>{cards}</body></html>'
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")


# ----------------------------------------------------------------------------
# 6. THE AGENT LOOP
# ----------------------------------------------------------------------------
def run(data_path: Path, request: str, out_dir: Path) -> Path:
    """Run the visualization agent. Returns the path of the index.html gallery.
    (Called by main.py; the command-line entry point is main() below.)"""
    data_path, out_dir = Path(data_path), Path(out_dir)
    check_file(data_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": f"REQUEST: {request}\n\nDATA PROFILE:\n{quick_profile(data_path)}"}]
    charts: dict = {}

    def finish_now(summary: str) -> None:
        write_gallery(out_dir, charts, summary)
        print(f"\nDone. Open {out_dir / 'index.html'} in your browser.")

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step}/{MAX_STEPS} ---")
        response = client.messages.create(
            model=MODEL, max_tokens=3000, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(block.text)
            elif block.type == "tool_use":
                if block.name == "run_python":
                    result = run_python(block.input["code"], data_path, out_dir)
                    print(f"[inspect]\n{result}")
                    log(out_dir, {"step": step, "tool": "run_python", "code": block.input["code"], "result": result})
                elif block.name == "create_chart":
                    print(f"[chart] {block.input.get('filename')}: {block.input.get('purpose')}")
                    result = create_chart(block.input, data_path, out_dir, charts)
                    log(out_dir, {"step": step, "tool": "create_chart", "input": block.input,
                                  "ok": isinstance(result, list)})
                elif block.name == "finish":
                    finish_now(block.input["summary_markdown"])
                    return out_dir / "index.html"
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            messages.append({"role": "user", "content": "Continue, or call finish if you are done."})

    # Safety net: step limit reached
    print("\nStep limit reached; finalizing with what exists.")
    finish_now("Stopped at the step limit. These are the charts created so far.")
    return out_dir / "index.html"


def main() -> None:
    """Standalone command-line entry point (same behaviour as before)."""
    if len(sys.argv) < 2:
        sys.exit('Usage: python data_visualization.py <file.csv|xlsx> ["what you want to see"]')
    data_path = Path(sys.argv[1])
    request = sys.argv[2] if len(sys.argv) > 2 else "Choose the most insightful charts for this data."
    out_dir = Path("output_viz") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run(data_path, request, out_dir)


if __name__ == "__main__":
    main()