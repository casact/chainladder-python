import json
import os
import re
import subprocess
import sys

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_lines(base):
    diff = subprocess.run(
        ["git", "diff", "-U0", f"{base}...HEAD", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif path and (match := HUNK.match(line)):
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def main(base):
    changed = changed_lines(base)
    violations = json.load(sys.stdin)
    root = os.getcwd()
    for violation in violations:
        path = os.path.relpath(violation["filename"], root)
        row = violation["location"]["row"]
        if row in changed.get(path, ()):
            message = f"{violation['code']} {violation['message']}"
            print(f"::warning file={path},line={row}::{message}")


if __name__ == "__main__":
    main(sys.argv[1])
