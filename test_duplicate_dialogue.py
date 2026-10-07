"""Test to verify duplicate dialogue handling."""

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter

# Create a paragraph with duplicate dialogue text
paragraph_text = '"Hello," said Harry. "Hello," said Ron.'

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
        print(f"Segment {seg.id}: replacing {repr(seg.text)} with {repr(marker)}")
        before = marked_paragraph
        marked_paragraph = marked_paragraph.replace(seg.text, marker, 1)
        print(f"  Before: {repr(before)}")
        print(f"  After:  {repr(marked_paragraph)}")
        quote_idx += 1

print()
print("Final marked paragraph:")
print(repr(marked_paragraph))
print()

# The issue: both segments have text='"Hello,"'
# The first replace will replace the FIRST occurrence in the paragraph
# But if segments are processed in order, this should work correctly
# However, if the order is wrong, it could fail
