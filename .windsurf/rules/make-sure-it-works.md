---
trigger: always_on
---

Always ensure scripts run end-to-end and produce expected results.
• Explain every method/function/class used (docstrings + inline comments where needed), especially parameters, return types, side effects, and assumptions.
• NO HARDCODING (paths, IDs, credentials, constants). Use config files, environment variables, or CLI args.
• NO SHORTCUTS. NO “sample-only” runs for simplicity/laziness. Use full, representative data unless a smaller subset is explicitly justified in the project plan.
• Execute the script yourself, validate outputs, and confirm behavior matches the spec.
• If you fix anything, update the explanations/comments accordingly.
• Read and analyze results; confirm they meet expectations. If not: record findings, propose a better approach, fix, re-run, re-analyze. Iterate until results are reasonable and satisfactory.
• Document outcomes and changes in the core docs only: README.md, PROJECT_STATUS.md, PERFECT_PROJECT_ROADMAP.md (checklist), and COMPLETE_PIPELINE_ANALYSIS.md.