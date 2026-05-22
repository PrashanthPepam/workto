"""Unit tests for KBStore — no database, no HTTP, no LLM."""

import pytest

from app.kb.store import KBAccessError, KBFileNotFoundError, KBStore


@pytest.fixture
def store(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "alpha.txt").write_text("Alpha Title\nAlpha body text.", encoding="utf-8")
    (kb / "beta.txt").write_text("Beta Title\nBeta body text.", encoding="utf-8")
    (kb / "gamma.txt").write_text("\n\n  \nGamma after blanks", encoding="utf-8")
    return KBStore(str(kb))


# ── list_files ─────────────────────────────────────────────────────────────────


def test_list_files_returns_all_txt(store) -> None:
    entries = store.list_files()
    names = [e["filename"] for e in entries]
    assert names == ["alpha.txt", "beta.txt", "gamma.txt"]  # sorted


def test_list_files_includes_summary(store) -> None:
    entries = store.list_files()
    assert entries[0]["summary"] == "Alpha Title"
    assert entries[1]["summary"] == "Beta Title"


def test_list_files_skips_blank_lines_for_summary(store) -> None:
    entries = store.list_files()
    gamma = next(e for e in entries if e["filename"] == "gamma.txt")
    assert gamma["summary"] == "Gamma after blanks"


def test_list_files_empty_dir(tmp_path) -> None:
    store = KBStore(str(tmp_path))
    assert store.list_files() == []


def test_list_files_ignores_non_txt(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("# Notes", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "doc.txt").write_text("Doc", encoding="utf-8")
    store = KBStore(str(tmp_path))
    names = [e["filename"] for e in store.list_files()]
    assert names == ["doc.txt"]


def test_list_files_summary_capped_at_200_chars(tmp_path) -> None:
    long_line = "x" * 300
    (tmp_path / "long.txt").write_text(long_line, encoding="utf-8")
    store = KBStore(str(tmp_path))
    summary = store.list_files()[0]["summary"]
    assert len(summary) <= 200


# ── read_file ──────────────────────────────────────────────────────────────────


def test_read_file_returns_full_content(store) -> None:
    content = store.read_file("alpha.txt")
    assert "Alpha Title" in content
    assert "Alpha body text." in content


def test_read_file_not_found_raises(store) -> None:
    with pytest.raises(KBFileNotFoundError):
        store.read_file("nonexistent.txt")


def test_read_file_path_traversal_blocked(store) -> None:
    with pytest.raises((KBAccessError, KBFileNotFoundError)):
        store.read_file("../secret.txt")


def test_read_file_absolute_path_blocked(store, tmp_path) -> None:
    # Create a file outside the KB root
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises((KBAccessError, KBFileNotFoundError)):
        store.read_file(str(outside))
