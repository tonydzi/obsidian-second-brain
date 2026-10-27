"""Link-graph reorder behind OBSIDIAN_SEARCH_PPR (discussion #342).

Personalized PageRank over the [[wikilink]] graph, seeded from the top of the
current ranking and fused back by RRF, with the first positions pinned. These
tests pin the contract the discussion set: off by default and byte-identical
when off, a reorder that never adds a note, a head a well-linked hub cannot
displace, and a failure that falls back to the search order.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))

import vault_ops  # noqa: E402

CANON = "wiki/concepts/Zorbix.md"
# The head size promised in #342, written out so a changed constant fails here.
HEAD = 5


def _write(v: Path, rel: str, body: str) -> None:
    p = v / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    """One short canonical note and eight longer notes that mention its term
    more often and link to it: the lexical ranker puts the canonical note below
    the top 5, the shape the synthetic keyword set is built around."""
    v = tmp_path / "vault"
    _write(v, CANON, "---\ntype: concept\n---\n\nZorbix is a windowing method.\n")
    for i in range(8):
        _write(
            v, f"wiki/daily/zorbix-log-{i}.md",
            "---\ntype: note\n---\n\n"
            + "\n".join(f"- zorbix run {j} was fine, see [[Zorbix]]" for j in range(4 + i))
            + "\n",
        )
    for i in range(4):
        _write(v, f"wiki/notes/filler-{i}.md", "---\ntype: note\n---\n\nnothing here\n")
    monkeypatch.setenv(vault_ops._VAULT_ENV, str(v))
    yield v


def _load(monkeypatch, on: bool):
    if on:
        monkeypatch.setenv("OBSIDIAN_SEARCH_PPR", "1")
    else:
        monkeypatch.delenv("OBSIDIAN_SEARCH_PPR", raising=False)
    return importlib.reload(vault_ops)


@pytest.fixture(autouse=True)
def _reload_back(monkeypatch):
    yield
    monkeypatch.delenv("OBSIDIAN_SEARCH_PPR", raising=False)
    importlib.reload(vault_ops)


def _paths(results):
    return [r["path"] for r in results]


def test_off_by_default_and_reorder_never_runs(vault, monkeypatch):
    mod = _load(monkeypatch, on=False)
    assert mod._PPR_ENABLED is False

    def boom(*a, **k):
        raise AssertionError("reorder ran with the switch off")

    monkeypatch.setattr(mod, "_ppr_reorder", boom)
    out = mod.search("zorbix", limit=10)
    assert CANON in _paths(out)
    # Links are not even parsed when off: the default path pays nothing.
    entry = next(iter(mod._SCAN_CACHE.values()))[1]
    assert entry[6] is None


def test_switch_on_with_a_neutral_reorder_matches_switch_off(vault, monkeypatch):
    """The deeper pool and the extra branch must not move anything by
    themselves: only the PPR vote may change the order."""
    off = _paths(_load(monkeypatch, on=False).search("zorbix", limit=10))
    mod = _load(monkeypatch, on=True)
    monkeypatch.setattr(mod, "_ppr_reorder", lambda results, *a, **k: results[:a[2]])
    assert _paths(mod.search("zorbix", limit=10)) == off


def test_canonical_note_moves_up_but_head_stays(vault, monkeypatch):
    off = _paths(_load(monkeypatch, on=False).search("zorbix", limit=10))
    assert off.index(CANON) >= HEAD, "fixture must bury the canonical note"
    on = _paths(_load(monkeypatch, on=True).search("zorbix", limit=10))
    assert on[: HEAD] == off[: HEAD]
    assert on.index(CANON) < off.index(CANON)
    assert sorted(on) == sorted(off), "a reorder, not a different result set"


def test_without_the_head_guard_the_hub_enters_the_top(vault, monkeypatch):
    """The other half of the guard test: with head=0 the canonical note does
    push into the top 5, so the pinned head is what keeps the textual top."""
    off = _paths(_load(monkeypatch, on=False).search("zorbix", limit=10))
    mod = _load(monkeypatch, on=True)
    links, stems = {}, {}
    for md in mod._iter_notes(vault):
        rel = md.relative_to(vault).as_posix()
        entry = mod._scan_entry(md, vault)
        stems.setdefault(mod._nfc(md.stem).lower(), rel)
        if entry[6]:
            links[rel] = entry[6]
    base = [{"path": p} for p in off]
    head = HEAD
    assert mod._PPR_HEAD == HEAD
    unguarded = _paths(mod._ppr_reorder(base, links, stems, 10, head=0))
    guarded = _paths(mod._ppr_reorder(base, links, stems, 10, head=head))
    assert CANON in unguarded[:head]
    assert guarded[:head] == off[:head]


def test_a_voted_note_outside_the_results_is_not_added(monkeypatch):
    results = [{"path": f"n{i}.md"} for i in range(8)]
    links = {f"n{i}.md": frozenset({"outside"}) for i in range(8)}
    stems = {f"n{i}": f"n{i}.md" for i in range(8)}
    stems["outside"] = "outside.md"
    out = vault_ops._ppr_reorder(results, links, stems, 8)
    assert _paths(out) == _paths(results)


def test_failure_falls_back_to_the_search_order(capsys):
    results = [{"path": f"n{i}.md"} for i in range(8)]
    out = vault_ops._ppr_reorder(results, {"n0.md": 42}, {}, 6)
    assert _paths(out) == _paths(results[:6])
    assert "reorder failed" in capsys.readouterr().err


def test_dangling_mass_returns_to_the_seed():
    # a -> hub, hub has no outgoing link: its mass restarts at the seed, not
    # uniformly, and total mass is conserved.
    mass = vault_ops._personalized_pagerank({"a": ["hub"]}.get, {"a": 1.0})
    assert abs(sum(mass.values()) - 1.0) < 1e-9
    assert set(mass) == {"a", "hub"}
    assert mass["hub"] > 0


def test_rrf_ties_go_to_the_earlier_position():
    assert vault_ops._rrf_positions([0, 1, 2], [2]) == [2, 0, 1]
    assert vault_ops._rrf_positions([0, 1], []) == [0, 1]
