"""STATUS.md: a table of the last 60 nightly runs, rewritten from the status.json history beside it."""
import json
import os

KEEP = 60
COLUMNS = ("Date", "Seq", "Contributors", "Promoted rows", "New sets", "Regression top-1 (before -> after)", "Exam top-1 (before -> after)", "Shipped")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pair(score, with_n=False):
    if not score:
        return "-"
    text = f"{score['before']} -> {score['after']}"
    return f"{text} (n={score['n']})" if with_n and "n" in score else text


def _row(r):
    sets = ", ".join(r.get("newSets") or []) or "-"
    cells = (r["date"], r["seq"], r.get("contributors", 0), r.get("promoted", 0), sets,
             _pair(r.get("regression")), _pair(r.get("exam"), with_n=True), "yes" if r.get("shipped") else "no")
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _history(path):
    try:
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
        return hist if isinstance(hist, list) else []
    except (OSError, ValueError):
        return []


def write(path, run_record):
    """Append run_record to status.json (next to path), keep the last 60, and rewrite path, newest run first."""
    folder = os.path.dirname(os.path.abspath(path))
    history_path = os.path.join(folder, "status.json")
    history = (_history(history_path) + [run_record])[-KEEP:]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    lines = ["# Learning status", "", f"Last {KEEP} nightly runs, newest first. Scores are top-1 on the fixed regression set and the community exam.", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += [_row(r) for r in reversed(history)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
