from pathlib import Path
import subprocess


REQUIRED_FILES = [
    "streamlit_starter_app.py",
    ".streamlit/secrets.example.toml",
    "README_STREAMLIT_STARTER.md",
    ".gitignore",
]

SENSITIVE_FILES = [
    ".env",
    ".streamlit/secrets.toml",
]

PREVIEW_FILES = [
    "alpaca_preview_orders_hybrid/proposed_orders.csv",
    "alpaca_preview_orders_hybrid/current_positions.csv",
    "alpaca_preview_orders_hybrid/model_scores.csv",
    "alpaca_preview_orders_hybrid/open_orders.csv",
]

GENERATED_FOLDERS = [
    "logs",
    "data_cache",
    "alpaca_preview_orders_hybrid",
    "alpaca_submitted_orders_hybrid",
    "ticker_ranking_results",
    "trade_log_results",
    "hybrid_trade_log_results",
]


def git_tracked_files():
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.splitlines())


def main():
    tracked = git_tracked_files()

    print("")
    print("===== STARTER KIT SAFETY CHECK =====")
    print("")

    print("Required starter files:")
    for file in REQUIRED_FILES:
        if Path(file).exists():
            print("  OK      " + file)
        else:
            print("  MISSING " + file)

    print("")
    print("Sensitive files should NOT be tracked:")
    for file in SENSITIVE_FILES:
        is_tracked = file in tracked
        exists = Path(file).exists()

        if is_tracked:
            print("  PROBLEM " + file + " is tracked by Git")
        elif exists:
            print("  OK      " + file + " exists locally but is not tracked")
        else:
            print("  OK      " + file + " is not present and not tracked")

    print("")
    print("Preview files used by read-only dashboard:")
    for file in PREVIEW_FILES:
        if Path(file).exists():
            print("  OK      " + file)
        else:
            print("  MISSING " + file)

    print("")
    print("Generated output folders should generally stay untracked:")
    for folder in GENERATED_FOLDERS:
        prefix = folder + "/"
        tracked_inside = [f for f in tracked if f.startswith(prefix)]

        if tracked_inside:
            print("  WARNING " + folder + "/ has tracked files")
        else:
            print("  OK      " + folder + "/ is not tracked")

    print("")
    print("Safety check complete.")
    print("")


if __name__ == "__main__":
    main()
