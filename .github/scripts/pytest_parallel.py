"""
Runs pytest in parallel (xdist --dist=loadfile) and reformats the output
to match the sequential per-file dots format.

With `rich` installed: live progress bars, a counts table, and colored output.
Without `rich`: plain-text \r progress lines and uncolored dots.

Coverage is enabled by default (--cov=chainladder --cov-report=term-missing).
Any extra arguments are forwarded to pytest after the defaults.

Usage:
    python .github/scripts/pytest_parallel.py [extra pytest args...]
    python .github/scripts/pytest_parallel.py --cov-report=xml
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import struct
import sys
import shutil
import subprocess
import termios
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Iterator

try:
    from rich.console import (
        Console,
        Group
    )
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        MofNCompleteColumn,
        Task,
        TextColumn,
        TimeElapsedColumn
    )
    from rich.spinner import Spinner as _Spinner
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if TYPE_CHECKING:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        Task,
        TextColumn,
        TimeElapsedColumn
    )
    from rich.spinner import Spinner as _Spinner
    from rich.table import Table
    from rich.text import Text

# -- constants ----------------------------------------------------------------

STATUS_CHAR = {
    "PASSED":  ".",
    "FAILED":  "F",
    "ERROR":   "E",
    "SKIPPED": "s",
    "XFAILED": "x",
    "XPASSED": "X",
}

STATUS_STYLE = {
    ".": "green",
    "F": "bold red",
    "E": "bold red",
    "s": "yellow",
    "x": "yellow",
    "X": "yellow",
}

# (dict-key, display-label, base-color)
COUNTS_SPEC = [
    ("PASSED",  "Passed",  "green"),
    ("FAILED",  "Failed",  "red"),
    ("ERROR",   "Error",   "red"),
    ("SKIPPED", "Skipped", "yellow"),
    ("XFAILED", "XFailed", "yellow"),
    ("XPASSED", "XPassed", "yellow"),
]

# -- regexes ------------------------------------------------------------------

# [gw0] [  x%] PASSED chainladder/core/tests/test_foo.py::test_bar[param]
RESULT_RE = re.compile(
    r"^\[gw\d+]\s+\[\s*\d+%]\s+"
    r"(PASSED|FAILED|ERROR|SKIPPED|XFAILED|XPASSED)\s+(\S+)"
)
# initialized: x/n workers  /  collecting: x/n workers  /  created: n/n workers
WORKER_PHASE_RE = re.compile(
    r"(?:initialized|collecting|ready|created):\s+(\d+)/(\d+) workers"
)
# n workers [t items]
TOTAL_RE = re.compile(r"(\d+) workers \[(\d+) items]")
# Bare test-start announcement: path/to/test.py::test_name[param]
ANNOUNCEMENT_RE = re.compile(r"^\S+::")
# === N passed, M skipped ... ===
SUMMARY_LINE_RE = re.compile(r"^=+ .* =+$")
OUTCOME_TOKEN_RE = re.compile(
    r"(\d+) (passed|failed|error(?:s|ed)?|skipped|warnings?|xfailed|xpassed)"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_STRIP_EQUALS_RE = re.compile(r"^=+\s*|\s*=+$")

_term_width = shutil.get_terminal_size(fallback=(120, 40)).columns

# -- rich-only module-level setup ---------------------------------------------

if HAS_RICH:
    class _StatusSpinnerColumn(ProgressColumn):
        """Animated spinner while running; checkmark (green) or x (red) when finished."""
        def __init__(self) -> None:
            self._spinner = _Spinner("dots")
            super().__init__()

        def render(self, task: Task) -> Text:
            """Render the column cell for a given task.

            Parameters
            ----------
            task : Task
                The Rich progress task being rendered.

            Returns
            -------
            Text
                A green ``✓`` or red ``✗`` when the task is finished,
                depending on the ``success`` field; an animated blue dots
                spinner while the task is still running.
            """
            if task.finished:
                if task.fields.get("success", True):
                    return Text("✓", style="bold green")
                return Text("✗", style="bold red")
            spinner_text: Text = self._spinner.render(task.get_time())  # type: ignore[assignment]
            spinner_text.stylize("bold blue")
            return spinner_text

    OUTCOME_STYLE: dict[str, str] = {
        "passed":  "bold green",
        "failed":  "bold red",
        "error":   "bold red",
        "skipped": "bold yellow",
        "warning": "bold yellow",
        "xfailed": "bold yellow",
        "xpassed": "bold yellow",
    }
    console = Console(highlight=False, width=_term_width)

# -- shared helpers -----------------------------------------------------------

def _decode(raw: bytes) -> str:
    """Decode bytes to a string, stripping ANSI escape codes.

    Parameters
    ----------
    raw : bytes
        Raw bytes from a PTY or subprocess stream.

    Returns
    -------
    str
        Decoded string with ANSI escape sequences removed and trailing
        whitespace stripped.
    """
    return ANSI_ESCAPE_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()


def _read_pty_lines(fd: int) -> Iterator[tuple[str, bool]]:
    """Read pytest's output from the PTY and yield it one line at a time.

    Each yielded item is a ``(line, is_overwrite)`` pair. ``is_overwrite``
    indicates whether the line is a temporary in-place status update (ending
    in ``\\r``, which overwrites the current terminal line) or a permanent
    output line (ending in ``\\n``). This distinction lets callers handle
    xdist's running progress ticks separately from real result lines.

    Parameters
    ----------
    fd : int
        File descriptor for the PTY main end.

    Yields
    ------
    tuple[str, bool]
        ``(line, is_overwrite)`` where ``is_overwrite`` is ``True`` for
        ``\\r``-terminated lines and ``False`` for ``\\n``-terminated lines.
    """
    buf = b""
    while True:
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        buf += data
        # Normalize Windows-style CRLF first so a lone \r (xdist in-place
        # update) is never confused with the \r inside a \r\n pair.
        buf = buf.replace(b"\r\n", b"\n")
        while True:
            r = buf.find(b"\r")
            n = buf.find(b"\n")
            if r == -1 and n == -1:
                break
            # Yield whichever terminator appears first; \r lines are in-place
            # overwrites, \n lines are permanent output.
            if n == -1 or (r != -1 and r < n):
                text = _decode(buf[:r]); buf = buf[r + 1:]
                if text.strip():
                    yield text, True
            else:
                text = _decode(buf[:n]); buf = buf[n + 1:]
                if text.strip():
                    yield text, False
    if buf:
        text = _decode(buf)
        if text.strip():
            yield text, False


def _worker_phase(
    lines_iter: Iterator[tuple[str, bool]],
    header: list[str],
    on_update: Callable[[int, int], None],
) -> int:
    """Consume lines until the ``N workers [M items]`` line and return M.

    Parameters
    ----------
    lines_iter : Iterator[tuple[str, bool]]
        Line iterator produced by ``_read_pty_lines``.`
    header : list[str]
        Accumulator for lines emitted before the test run begins.
    on_update : Callable[[int, int], None]
        Callback invoked with ``(ready_workers, total_workers)`` whenever a
        worker-phase progress line is parsed.

    Returns
    -------
    int
        Total number of test items collected, as reported by xdist.
        Returns ``0`` if the iterator is exhausted before the summary line
        appears.
    """
    last_worker_total = 0
    max_n = 0
    for line, is_overwrite in lines_iter:
        wm = WORKER_PHASE_RE.search(line)
        if wm:
            n, m = int(wm.group(1)), int(wm.group(2))
            last_worker_total = m
            if n > max_n:
                max_n = n
                on_update(n, m)
        if not is_overwrite:
            tm = TOTAL_RE.search(line)
            if tm:
                # xdist never emits the final N/N tick - force the bar to 100%.
                if last_worker_total:
                    on_update(last_worker_total, last_worker_total)
                header.append(line)
                return int(tm.group(2))
            # xdist emits bare "path::test[param]" lines during collection;
            # skip them so they don't pollute the header output.
            if not ANNOUNCEMENT_RE.match(line):
                header.append(line)
    return 0


def _test_phase(
    lines_iter: Iterator[tuple[str, bool]],
    header: list[str],
    footer: list[str],
    results: dict[str, list[str]],
    file_order: list[str],
    on_result: Callable[[str], None],
) -> None:
    """Consume result lines and call ``on_result`` for each completed test.

    Parameters
    ----------
    lines_iter : Iterator[tuple[str, bool]]
        Line iterator produced by ``_read_pty_lines``.
    header : list[str]
        Accumulator for lines emitted before the first result line.
    footer : list[str]
        Accumulator for lines emitted after the first result line (summary,
        warnings, etc.).
    results : dict[str, list[str]]
        Mapping of file path to the list of single-character status codes
        (e.g. ``"."``, ``"F"``) collected for that file.
    file_order : list[str]
        Ordered list of file paths in the order they first appear in the output.
    on_result : Callable[[str], None]
        Callback invoked with the raw status string (e.g. ``"PASSED"``) for
        each test result line parsed.

    Returns
    -------
    None
    """
    in_results = False
    for line, is_overwrite in lines_iter:
        if is_overwrite:
            continue
        m = RESULT_RE.match(line)
        if m:
            in_results = True
            status, node_id = m.group(1), m.group(2)
            file_path = node_id.split("::")[0]
            if file_path not in results:
                file_order.append(file_path)
            results[file_path].append(STATUS_CHAR[status])
            on_result(status)
        elif not ANNOUNCEMENT_RE.match(line):
            (footer if in_results else header).append(line)

# -- plain-text output --------------------------------------------------------

def _print_results_plain(
    header: list[str],
    file_order: list[str],
    results: dict[str, list[str]],
    footer: list[str],
) -> None:
    """Print test results in plain-text format without Rich.

    Parameters
    ----------
    header : list[str]
        Lines to print before the per-file result rows.
    file_order : list[str]
        Ordered list of file paths that produced results.
    results : dict[str, list[str]]
        Mapping of file path to the list of single-character status codes.
    footer : list[str]
        Lines to print after the per-file result rows (summary, warnings, etc.).

    Returns
    -------
    None
    """
    total_collected = sum(len(v) for v in results.values())
    for line in header:
        print(line)
    cumulative = 0
    for file_path in sorted(file_order):
        chars = results[file_path]
        cumulative += len(chars)
        pct = int(cumulative / total_collected * 100) if total_collected else 0
        pct_str = f"[{pct:3d}%]"
        dots = "".join(chars)
        padding = max(1, _term_width - len(file_path) - 1 - len(dots) - len(pct_str))
        print(file_path + " " + dots + " " * padding + pct_str)
    for line in footer:
        print(line)

# -- rich output --------------------------------------------------------------

if HAS_RICH:
    def _make_counts_table(counts: dict[str, int]) -> Table:
        """Build a Rich grid table showing current pass/fail/skip/etc. counts.

        Parameters
        ----------
        counts : dict[str, int]
            Mapping of status key (e.g. ``"PASSED"``) to its current count.

        Returns
        -------
        Table
            A single-row Rich grid table with one colored label per outcome.
        """
        table = Table.grid(padding=(0, 3))  # type: ignore[union-attr]
        table.add_row(*[
            Text.assemble((label + ": ", color), (str(counts[key]), "bold " + color))
            for key, label, color in COUNTS_SPEC
        ])
        return table

    def _make_progress() -> Progress:
        """Create a Rich Progress bar configured for test-run tracking.

        Returns
        -------
        Progress
            A Rich ``Progress`` instance with a spinner column, description,
            bar, M-of-N counter, and elapsed-time columns.
        """
        return Progress(
            _StatusSpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

    def _bordered_text(inner_text: Text, color: str) -> Text:
        """Surround ``inner_text`` with ``=`` padding to fill the terminal width.

        Parameters
        ----------
        inner_text : Text
            The Rich ``Text`` object to center between equals-sign borders.
        color : str
            Rich color/style string applied to the ``=`` border characters.

        Returns
        -------
        Text
            A new Rich ``Text`` with the inner text flanked by ``=`` borders.
        """
        w = len(inner_text.plain) + 2
        left = (_term_width - w) // 2
        right = _term_width - w - left
        row = Text()
        row.append("=" * left + " ", style=f"bold {color}")
        row.append_text(inner_text)
        row.append(" " + "=" * right, style=f"bold {color}")
        return row

    def _dominant_color(inner: str) -> str:
        """Determine the display color from the outcome tokens in a summary line.

        Parameters
        ----------
        inner : str
            Plain-text content of a pytest summary line, without ``=`` borders.

        Returns
        -------
        str
            ``"red"`` if any failures or errors are present, ``"yellow"`` if
            only skips/warnings/xfailed, otherwise ``"green"``.
        """
        words = {m.group(2).rstrip("s") for m in OUTCOME_TOKEN_RE.finditer(inner)}
        if words & {"failed", "error"}:
            return "red"
        if words & {"skipped", "warning", "xfailed"}:
            return "yellow"
        return "green"

    def _render_section_header(line: str) -> Text:
        """Render a pytest ``=== section header ===`` line as a Rich ``Text`` object.

        Parameters
        ----------
        line : str
            A raw pytest section-header line (e.g. ``"=== warnings summary ===``").

        Returns
        -------
        Text
            A Rich ``Text`` with the section title centered between colored
            ``=`` borders, styled according to whether it is a warning or error.
        """
        inner = _STRIP_EQUALS_RE.sub("", line).strip()
        lower = inner.lower()
        if "warn" in lower:
            color = "yellow"
        elif "fail" in lower or "error" in lower:
            color = "red"
        else:
            color = "white"
        return _bordered_text(Text(inner, style=f"bold {color}"), color)

    def _render_summary_line(line: str) -> Text:
        """Render a pytest ``=== N passed, M failed ... ===`` line as a Rich ``Text``.

        Each outcome token is colored individually according to ``OUTCOME_STYLE``.

        Parameters
        ----------
        line : str
            A raw pytest final-summary line.

        Returns
        -------
        Text
            A Rich ``Text`` with per-token coloring centered between ``=``
            borders in the dominant outcome color.
        """
        inner = _STRIP_EQUALS_RE.sub("", line).strip()
        color = _dominant_color(inner)
        inner_text = Text()
        pos = 0
        for m in OUTCOME_TOKEN_RE.finditer(inner):
            if m.start() > pos:
                inner_text.append(inner[pos: m.start()], style=color)
            key = m.group(2).rstrip("s")
            inner_text.append(m.group(0), style=OUTCOME_STYLE.get(key, color))
            pos = m.end()
        if pos < len(inner):
            inner_text.append(inner[pos:], style=color)
        return _bordered_text(inner_text, color)

    def _print_results_rich(
        header: list[str],
        file_order: list[str],
        results: dict[str, list[str]],
        footer: list[str],
    ) -> None:
        """Print test results using Rich markup with per-character status coloring.

        Parameters
        ----------
        header : list[str]
            Lines to print before the per-file result rows.
        file_order : list[str]
            Ordered list of file paths that produced results.
        results : dict[str, list[str]]
            Mapping of file path to the list of single-character status codes.
        footer : list[str]
            Lines to print after the per-file result rows (summary, warnings,
            etc.).

        Returns
        -------
        None
        """
        total_collected = sum(len(v) for v in results.values())
        for line in header:
            tm = TOTAL_RE.search(line)
            if tm:
                row = Text(f"{tm.group(1)} workers [")
                row.append(f"collected {tm.group(2)} items", style="bold")
                row.append("]")
                console.print(row)
            else:
                console.print(line, markup=False, highlight=False)
                if "scheduling tests via" in line:
                    console.print()
        cumulative = 0
        pct_color = "green"
        for file_path in sorted(file_order):
            chars = results[file_path]
            cumulative += len(chars)
            pct = int(cumulative / total_collected * 100) if total_collected else 0
            pct_str = f"[{pct:3d}%]"
            # Mirror pytest: escalate color based on worst result seen so far.
            for c in chars:
                if c in ("F", "E"):
                    pct_color = "red"
                elif c in ("s", "x", "X") and pct_color != "red":
                    pct_color = "yellow"
            padding = max(1, console.width - len(file_path) - 1 - len(chars) - len(pct_str))
            row = Text(file_path + " ")
            for c in chars:
                row.append(c, style=STATUS_STYLE.get(c, "white"))
            row.append(" " * padding)
            row.append(pct_str, style=pct_color)
            # no_wrap keeps the percentage badge right-aligned; wrapping would
            # break the fixed-width layout for long file paths.
            console.print(row, no_wrap=True)
        for line in footer:
            if SUMMARY_LINE_RE.match(line) and OUTCOME_TOKEN_RE.search(line):
                console.print(_render_summary_line(line))
            elif SUMMARY_LINE_RE.match(line):
                console.print(_render_section_header(line))
            else:
                console.print(line, markup=False, highlight=False)

# -- entry point --------------------------------------------------------------

def main() -> None:
    """
    Run pytest in parallel and reformat its output to sequential dots format.

    Launches pytest with ``-n auto --dist=loadfile`` inside a PTY (PseudoTerminal) so that
    color and terminal width are preserved, then re-streams the output grouped
    by file.  With ``rich`` installed, an animated progress bar and colored
    output are shown; otherwise plain ``\\r``-overwrite progress lines are used.

    Extra arguments passed on the command line are forwarded to pytest after
    the defaults.  The process exits with pytest's return code.

    Returns
    -------
    None
    """

    # Set up command-line arguments.
    cmd = [
        sys.executable, "-m", "pytest",
        "-n", "auto", "--dist=loadfile", "-v",
        "--cov=chainladder", "--cov-report=term-missing",
        *sys.argv[1:],
    ]

    # Establish ends of the PTY.
    # main_fd: file descriptor for the parent process governing the entire script.
    # child_fd: file descriptor for the child process, i.e., pytest.
    main_fd, child_fd = pty.openpty()
    # Tell pytest the PTY is as wide as the real terminal so its === headers fill correctly.
    winsize = struct.pack("HHHH", 40, _term_width, 0, 0)
    fcntl.ioctl(child_fd, termios.TIOCSWINSZ, winsize)
    # Run pytest with the child end of the PTY as both stdout and stderr so
    # that pytest believes it is writing to a real terminal (enabling color
    # and full-width === headers).
    proc = subprocess.Popen(cmd, stdout=child_fd, stderr=child_fd, close_fds=True)
    # Close the parent's copy of the child fd; if we keep it open, os.read()
    # on the main fd will never receive EOF after the child exits.
    os.close(child_fd)
    lines_iter = _read_pty_lines(main_fd)

    header: list[str] = []
    footer: list[str] = []
    results: dict[str, list[str]] = defaultdict(list)
    file_order: list[str] = []

    # Pretty formatting when user has rich installed.
    if HAS_RICH:
        # Collect workers.
        with _make_progress() as wp:
            worker_task = wp.add_task("Collecting workers...", total=None, success=True)
            total = _worker_phase(
                lines_iter, header,
                lambda n, m: wp.update(worker_task, completed=n, total=m),
            )

        # Execute tests.
        live_counts: dict[str, int] = {key: 0 for key, *_ in COUNTS_SPEC}
        test_progress = _make_progress()
        test_task = test_progress.add_task("Running tests...", total=total, success=True)

        # Render display.
        with Live(
            # Put counts table (Passed: 0  Failed: 0 ...) above the progress bar.
            Group(_make_counts_table(live_counts), test_progress),
            console=console,
            refresh_per_second=10,
        ) as live:
            # Tally test counts and advance progress bar.
            def _on_result_rich(status: str) -> None:
                live_counts[status] += 1
                test_progress.advance(test_task)
                live.update(Group(_make_counts_table(live_counts), test_progress))

            _test_phase(lines_iter, header, footer, results, file_order, _on_result_rich)
            has_failures = live_counts["FAILED"] + live_counts["ERROR"] > 0
            test_progress.update(test_task, success=not has_failures)
            live.update(Group(_make_counts_table(live_counts), test_progress))

        _print_results_rich(header, file_order, results, footer)

    # Use does not have rich installed generate plaintext output.
    else:
        # Collect workers.
        def _on_update_plain(n: int, m: int) -> None:
            print(f"\rCollecting workers... {n}/{m}  ", end="", flush=True)

        total = _worker_phase(lines_iter, header, _on_update_plain)
        print()  # end the \r line

        # Execute tests.
        completed = 0
        live_counts = {key: 0 for key, *_ in COUNTS_SPEC}

        def _on_result_plain(status: str) -> None:
            nonlocal completed
            live_counts[status] += 1
            completed += 1
            counts_str = "  ".join(
                f"{lbl}: {live_counts[k]}" for k, lbl, _ in COUNTS_SPEC
            )
            print(f"\r{counts_str}  [{completed}/{total}]  ", end="", flush=True)

        _test_phase(lines_iter, header, footer, results, file_order, _on_result_plain)
        print()  # end the \r line

        _print_results_plain(header, file_order, results, footer)

    os.close(main_fd)
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
