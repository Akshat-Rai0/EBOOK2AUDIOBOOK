"""Test to verify spacing preservation in the attribution stage."""

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter
from ebook2audiobook.models.book import Book, Chapter, Paragraph

# Create a paragraph with multiple spaces between quote and narration
paragraph_text = '"Hello,"  said Harry.  "How are you?"  asked Ron.'

# Simulate what the segmenter does
segmenter = DialogueSegmenter()
segments = segmenter.split_paragraph(
    paragraph_id="c00-p000",
    chapter_index=0,
    text=paragraph_text,
)

print("Original paragraph text:")
print(repr(paragraph_text))
print()

print("Segments created:")
for seg in segments:
    print(f"  {seg.id}: kind={seg.kind}, text={repr(seg.text)}")
print()

# Now simulate what the attribution stage does
marked_paragraph = paragraph_text
quote_idx = 0
for seg in segments:
    if seg.kind == "dialogue" and seg.speaker_id == "unknown":
        quote_id = f"Q{quote_idx + 1}"
        marker = f"[{quote_id}]"
        print(f"Trying to replace {repr(seg.text)} with {repr(marker)}")
        marked_paragraph = marked_paragraph.replace(seg.text, marker, 1)
        quote_idx += 1

print()
print("Marked paragraph result:")
print(repr(marked_paragraph))
print()

# Check if the replacement worked
if "[Q1]" in marked_paragraph and "[Q2]" in marked_paragraph:
    print("SUCCESS: Markers were inserted correctly")
else:
    print("FAILURE: Markers were NOT inserted correctly")
    print("This is because seg.text is stripped but paragraph.text has original spacing")
