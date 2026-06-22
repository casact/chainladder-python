#!/usr/bin/env bash
# Appends a pyright --verifytypes text report to $GITHUB_STEP_SUMMARY,
# wrapped in a collapsible <details> block. GitHub caps each step's summary
# at 1MiB and silently drops the whole upload if exceeded, so the report is
# truncated to a safe byte budget with a note when that happens, rather than
# risking the entire block vanishing.
set -euo pipefail

report_file="$1"
label="$2"
max_bytes=900000

total_bytes="$(wc -c < "$report_file")"

{
  echo "<details><summary>Full pyright --verifytypes output (${label})</summary>"
  echo ""
  echo '```text'
  head -c "$max_bytes" "$report_file"
  if [ "$total_bytes" -gt "$max_bytes" ]; then
    echo ""
    echo "... (truncated: ${total_bytes} bytes total, GITHUB_STEP_SUMMARY caps each step at 1MiB)"
  fi
  echo '```'
  echo "</details>"
} >> "$GITHUB_STEP_SUMMARY"
