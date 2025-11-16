---
trigger: always_on
---

Do not execute scripts directly in the terminal (e.g., .venv/bin/python -c "…").
Forbidden: python -c "…", here-docs (cat <<'PY' | python), echo '…' | python, multi-line shell one-liners, ad-hoc REPL runs that write files or change state.
Required: Always put code in a tracked .py file and run it (python path/to/script.py or python -m package.module). No exceptions.
If something “quick” is needed, still create a .py file in the repo (e.g., scripts/quick_checks/…py) instead of inline terminal code.