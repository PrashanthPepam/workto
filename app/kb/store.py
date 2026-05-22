"""
Knowledge base file store.

Provides two operations that map directly to the two OpenAI tools:

  list_files()        → [{filename, summary}]  for every .txt file.
                        Used as a lightweight index so the LLM can pick
                        relevant files without loading any content.

  read_file(filename) → full text of one file.
                        A path-traversal guard prevents escaping the KB root.

Neither method is async: reads are tiny (a few KB each) and the
latency bottleneck is always the LLM round-trip, not the local disk.

Sentinel exceptions let callers produce informative tool-result strings
instead of crashing the agent loop.
"""

from pathlib import Path


class KBFileNotFoundError(Exception):
    pass


class KBAccessError(Exception):
    pass


class KBStore:
    def __init__(self, knowledge_dir: str) -> None:
        # Resolve once so every subsequent check is against a stable absolute path.
        self.root = Path(knowledge_dir).resolve()

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_files(self) -> list[dict[str, str]]:
        """Return [{filename, summary}] for every .txt file, sorted by name."""
        entries = []
        for path in sorted(self.root.glob("*.txt")):
            entries.append({"filename": path.name, "summary": self._first_line(path)})
        return entries

    def read_file(self, filename: str) -> str:
        """Return the full UTF-8 text of *filename*.

        Raises
        ------
        KBAccessError       if *filename* would escape the KB root directory.
        KBFileNotFoundError if *filename* does not exist inside the root.
        """
        # Resolve to absolute path and verify it stays within self.root.
        # This blocks traversals like "../../../etc/passwd" or absolute paths.
        target = (self.root / filename).resolve()
        if not str(target).startswith(str(self.root) + "/") and target != self.root:
            # On Windows the separator is \ — use os.path for portability.
            import os
            if not str(target).startswith(str(self.root) + os.sep):
                raise KBAccessError(f"Access denied: {filename!r}")

        if not target.is_file():
            raise KBFileNotFoundError(f"File not found in knowledge base: {filename!r}")

        return target.read_text(encoding="utf-8")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _first_line(path: Path) -> str:
        """Return the first non-empty line (≤200 chars) as a one-line summary."""
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        return stripped[:200]
        except OSError:
            pass
        return ""
