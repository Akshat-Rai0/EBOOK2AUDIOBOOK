"""Test to demonstrate the chunked quote attribution bug."""

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

print("=" * 80)
print("ISSUE: Chunked dialogue gets multiple markers")
print("=" * 80)
print()
print("Original paragraph text:")
print(repr(long_dialogue))
print()

print("Segments created:")
for seg in segments:
    print(f"  {seg.id}: kind={seg.kind}, text={repr(seg.text)}")
print()

# Count dialogue segments
dialogue_segments = [s for s in segments if s.kind == "dialogue"]
print(f"Number of dialogue segments: {len(dialogue_segments)}")
print()

# Simulate the attribution stage logic
marked_paragraph = long_dialogue
quote_id_map = {}
quote_idx = 0

for seg in segments:
    is_unknown_dialogue = (
        seg.kind == "dialogue"
        and seg.speaker_id == "unknown"
    )
    if is_unknown_dialogue:
        quote_id = f"Q{quote_idx + 1}"
        quote_id_map[quote_id] = seg
        marker = f"[{quote_id}]"
        marked_paragraph = marked_paragraph.replace(seg.text, marker, 1)
        quote_idx += 1

print("Marked paragraph sent to LLM:")
print(repr(marked_paragraph))
print()

print("Quote ID map:")
for quote_id, seg in quote_id_map.items():
    print(f"  {quote_id} -> {seg.id}: {repr(seg.text)}")
print()

print("=" * 80)
print("PROBLEM:")
print("=" * 80)
print("The original paragraph has ONE quote (spoken by Harry).")
print("But the attribution stage creates THREE markers (Q1, Q2, Q3).")
print("The LLM will be asked to attribute three separate quotes,")
print("when there is actually only one continuous quote.")
print()
print("This could lead to:")
print("  1. LLM assigning different speakers to Q1, Q2, Q3")
print("  2. Inconsistent attribution within a single quote")
print("  3. Confusion in the attribution results")
print()
print("EXPECTED BEHAVIOR:")
print("  - A single quote, even if chunked, should get ONE marker")
print("  - The marker should replace the ENTIRE quote, not each chunk")
print("=" * 80)
