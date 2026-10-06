"""Test regression: Mr. Dursley/Mrs. Dursley and Mr. Potter/Mrs. Potter must NOT merge."""

from ebook2audiobook.attribution.character_registry import CharacterRegistry


def test_mr_mrs_same_surname_do_not_merge():
    """Regression test: Mr. Dursley and Mrs. Dursley should not merge."""
    registry = CharacterRegistry()

    # Resolve Mr. Dursley
    registry.resolve("Mr. Dursley")
    # Resolve Mrs. Dursley
    registry.resolve("Mrs. Dursley")

    # They should remain separate entries
    names = registry.character_names()
    assert "Mr. Dursley" in names
    assert "Mrs. Dursley" in names
    assert len(names) == 2


def test_mr_mrs_potter_do_not_merge():
    """Regression test: Mr. Potter and Mrs. Potter should not merge."""
    registry = CharacterRegistry()

    registry.resolve("Mr. Potter")
    registry.resolve("Mrs. Potter")

    names = registry.character_names()
    assert "Mr. Potter" in names
    assert "Mrs. Potter" in names
    assert len(names) == 2


def test_harry_potter_merged_with_variants():
    """Harry Potter should merge with "Harry" and "Potter" (same person)."""
    registry = CharacterRegistry()

    registry.resolve("Harry Potter")
    registry.resolve("Harry")
    registry.resolve("Potter")

    # All should merge into one canonical name
    names = registry.character_names()
    assert len(names) == 1
    # The longest name should be canonical
    assert "Harry Potter" in names


def test_title_conflict_blocks_merge():
    """Titles (Mr., Mrs.) should block merge when surnames are the same."""
    registry = CharacterRegistry()

    # Mr. Smith and Mrs. Smith should not merge
    registry.resolve("Mr. Smith")
    registry.resolve("Mrs. Smith")

    names = registry.character_names()
    assert "Mr. Smith" in names
    assert "Mrs. Smith" in names
    assert len(names) == 2
