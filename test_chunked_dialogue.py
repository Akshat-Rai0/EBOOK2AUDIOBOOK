"""Test to verify chunked dialogue handling."""

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter

# Create a paragraph with a very long dialogue that will be chunked
long_dialogue = '"This is a very long piece of dialogue that will definitely exceed the 400 character limit when we include all the text in this quote. It should be split into multiple segments by the chunker. We need to see if the replacement logic handles this correctly." said Harry.'

# Simulate what the segmenter does
segmenter = DialogueSegmenter(max_chars=100)  # Force chunking
segments = segmenter.split_paragraph(
    paragraph_id="c00-p000",
    chapter_index=0,
    text=long_dialogue,
)

print("Original paragraph text:")
print(repr(long_dialogue))
print()

print("Segments created:")
for seg in segments:
    print(f"  {seg.id}: kind={seg.kind}, text={repr(seg.text)}")
print()

# Now simulate what the attribution stage does
marked_paragraph = long_dialogue
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
        if seg.text not in before:
            print(f"  WARNING: seg.text not found in marked_paragraph!")
        quote_idx += 1

print()
print("Final marked paragraph:")
print(repr(marked_paragraph))
print()

# The issue: if dialogue is chunked, the segment text is only a PART of the quote
# The replacement will fail because the full quote text doesn't match the chunk
