"""Tests for cast seed builder with fake NER."""


from ebook2audiobook.attribution.cast_builder import (
    CastSeedFile,
    build_cast_seed,
)
from ebook2audiobook.models.book import Book, Chapter, Paragraph


def test_build_cast_seed_with_fake_ner():
    """Test cast building with fake NER function."""

    def fake_ner(text: str) -> list[tuple[str, int]]:
        """Fake NER that counts Harry and Ron."""
        results = []
        if "Harry" in text:
            results.append(("Harry", text.count("Harry")))
        if "Ron" in text:
            results.append(("Ron", text.count("Ron")))
        return results

    # Create a synthetic book
    book = Book(
        id="test-book",
        title="Test Book",
        author="Test Author",
        source_path="/fake/test.txt",
        source_format="txt",
        chapters=[
            Chapter(
                id="c01",
                index=0,
                title="Chapter 1",
                paragraphs=[
                    Paragraph(
                        id="c01-p001",
                        text="Harry said hello.",
                        original_text="Harry said hello.",
                    ),
                    Paragraph(
                        id="c01-p002",
                        text="Ron agreed with Harry.",
                        original_text="Ron agreed with Harry.",
                    ),
                    Paragraph(
                        id="c01-p003",
                        text="Harry and Ron walked together.",
                        original_text="Harry and Ron walked together.",
                    ),
                ],
            )
        ],
    )

    seeds = build_cast_seed(book, min_mentions=2, ner_func=fake_ner)

    # Should find Harry (3 mentions) and Ron (2 mentions)
    assert len(seeds) == 2
    assert seeds[0].name == "Harry"
    assert seeds[0].mention_count == 3
    assert seeds[1].name == "Ron"
    assert seeds[1].mention_count == 2


def test_build_cast_seed_filters_by_min_mentions():
    """Test that min_mentions filter works."""

    def fake_ner(text: str) -> list[tuple[str, int]]:
        """Fake NER that counts characters."""
        results = []
        if "Harry" in text:
            results.append(("Harry", text.count("Harry")))
        if "Minor" in text:
            results.append(("Minor", text.count("Minor")))
        return results

    book = Book(
        id="test-book",
        title="Test Book",
        author="Test Author",
        source_path="/fake/test.txt",
        source_format="txt",
        chapters=[
            Chapter(
                id="c01",
                index=0,
                title="Chapter 1",
                paragraphs=[
                    Paragraph(
                        id="c01-p001",
                        text="Harry said hello.",
                        original_text="Harry said hello.",
                    ),
                    Paragraph(
                        id="c01-p002",
                        text="Harry agreed.",
                        original_text="Harry agreed.",
                    ),
                    Paragraph(
                        id="c01-p003",
                        text="Minor said something.",
                        original_text="Minor said something.",
                    ),
                ],
            )
        ],
    )

    # With min_mentions=2, only Harry should be included
    seeds = build_cast_seed(book, min_mentions=2, ner_func=fake_ner)
    assert len(seeds) == 1
    assert seeds[0].name == "Harry"

    # With min_mentions=1, both should be included
    seeds = build_cast_seed(book, min_mentions=1, ner_func=fake_ner)
    assert len(seeds) == 2


def test_build_cast_seed_with_manual_seed_file(tmp_path):
    """Test that manual seed file adds characters even with 0 mentions."""

    def fake_ner(text: str) -> list[tuple[str, int]]:
        """Fake NER that only finds Harry."""
        if "Harry" in text:
            return [("Harry", text.count("Harry"))]
        return []

    # Create manual seed file
    seed_file = tmp_path / "cast_seed.json"
    seed_data = CastSeedFile(
        characters=[
            {"name": "The Boy Who Lived", "aliases": ["Boy", "Chosen One"]},
            {"name": "Voldemort", "aliases": ["Dark Lord", "You-Know-Who"]},
        ]
    )
    seed_file.write_text(seed_data.model_dump_json(indent=2))

    book = Book(
        id="test-book",
        title="Test Book",
        author="Test Author",
        source_path="/fake/test.txt",
        source_format="txt",
        chapters=[
            Chapter(
                id="c01",
                index=0,
                title="Chapter 1",
                paragraphs=[
                    Paragraph(
                        id="c01-p001",
                        text="Harry said hello.",
                        original_text="Harry said hello.",
                    ),
                ],
            )
        ],
    )

    seeds = build_cast_seed(book, min_mentions=1, seed_file_path=seed_file, ner_func=fake_ner)

    # Should include Harry (from NER) and manual seeds
    names = [s.name for s in seeds]
    assert "Harry" in names
    assert "The Boy Who Lived" in names
    assert "Voldemort" in names

    # Check aliases
    voldemort_seed = next(s for s in seeds if s.name == "Voldemort")
    assert "Dark Lord" in voldemort_seed.aliases
    assert "You-Know-Who" in voldemort_seed.aliases


def test_build_cast_seed_sorts_by_mention_count():
    """Test that results are sorted by mention count descending."""

    def fake_ner(text: str) -> list[tuple[str, int]]:
        """Fake NER with different counts."""
        results = []
        if "Harry" in text:
            results.append(("Harry", text.count("Harry")))
        if "Ron" in text:
            results.append(("Ron", text.count("Ron")))
        if "Hermione" in text:
            results.append(("Hermione", text.count("Hermione")))
        return results

    book = Book(
        id="test-book",
        title="Test Book",
        author="Test Author",
        source_path="/fake/test.txt",
        source_format="txt",
        chapters=[
            Chapter(
                id="c01",
                index=0,
                title="Chapter 1",
                paragraphs=[
                    Paragraph(
                        id="c01-p001",
                        text="Harry Harry Harry.",
                        original_text="Harry Harry Harry.",
                    ),  # 3
                    Paragraph(
                        id="c01-p002",
                        text="Ron Ron.",
                        original_text="Ron Ron.",
                    ),  # 2
                    Paragraph(
                        id="c01-p003",
                        text="Hermione.",
                        original_text="Hermione.",
                    ),  # 1
                ],
            )
        ],
    )

    seeds = build_cast_seed(book, min_mentions=1, ner_func=fake_ner)

    assert len(seeds) == 3
    assert seeds[0].name == "Harry"
    assert seeds[0].mention_count == 3
    assert seeds[1].name == "Ron"
    assert seeds[1].mention_count == 2
    assert seeds[2].name == "Hermione"
    assert seeds[2].mention_count == 1
