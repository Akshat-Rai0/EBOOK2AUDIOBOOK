"""Test to verify text matching between segments and paragraph."""

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter

# Create a paragraph with trailing/leading spaces in the quote
paragraph_text = '  "Hello," said Harry.  "How are you?" asked Ron.  '

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

# Simulate the attribution stage logic
marked_paragraph = paragraph_text
quote_idx = 0

for seg in segments:
    is_unknown_dialogue = (
        seg.kind == "dialogue"
        and seg.speaker_id == "unknown"
    )
    if is_unknown_dialogue:
        quote_id = f"Q{quote_idx + 1}"
        marker = f"[{quote_id}]"
        print(f"Trying to replace {repr(seg.text)} with {repr(marker)}")
        print(f"  Is {repr(seg.text)} in {repr(marked_paragraph)}? {seg.text in marked_paragraph}")
        before_len = len(marked_paragraph)
        marked_paragraph = marked_paragraph.replace(seg.text, marker, 1)
        after_len = len(marked_paragraph)
        print(f"  Replacement successful? {before_len != after_len}")
        quote_idx += 1

print()
print("Final marked paragraph:")
print(repr(marked_paragraph))
