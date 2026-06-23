#!/usr/bin/env bash
# Appends a pyright --verifytypes text report to $GITHUB_STEP_SUMMARY,
# wrapped in a collapsible <details> block. GitHub caps each step's summary
# at 1MiB and silently drops the whole upload if exceeded, so the report is
# truncated to a safe byte budget with a note when that happens, rather than
# risking the entire block vanishing.
#
# This script is called twice per workflow run (PR head and PR base). The
# budget below is half of the 1MiB cap, not the full cap, so the two calls
# stay safely under the limit even in the worst case where they end up
# sharing one combined budget instead of each getting their own per-step
# allotment (true today, per GitHub's docs, but cheap to hedge against).
set -euo pipefail

report_file="$1"
label="$2"
max_bytes=450000

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
