# Labelling Guide for M3 Attribution Evaluation

**Purpose:** This guide explains how to hand-label speaker attribution for quote spans in public-domain text. Your labels become the gold standard against which we measure attribution accuracy.

**Important:** Never look at model or rule output while labeling. The tool must not show any system guesses—this would bias your labels.

---

## Label Set

### Character Names
- Use the canonical character name (e.g., "sherlock", "harry", "jonathan_harker")
- The tool maintains a running list of characters you've already labeled
- Press `n` to add a new character with aliases

### Special Labels

| Label | When to Use | Example |
|--------|-------------|---------|
| **NARRATOR_I** | First-person narrator (the "I" character in epistolary text) | Letters, diaries, first-person narration where "I" is the narrator |
| **GROUP** | Multiple speakers speaking together | "They all shouted," "The crowd cheered" |
| **UNKNOWN** | Genuinely undeterminable speaker | No contextual clues, ambiguous attribution |
| **NOT_SPEECH** | Scare quotes, titles, quoted words, thoughts, internal monologue | "The 'best' way," title: *Dracula*, thought: *I wonder...* |

---

## Labeling Policies

### Unnamed Speakers
- Use a descriptive placeholder if the character is unnamed (e.g., "the_old_man", "the_stranger")
- Be consistent: if you use "the_old_man" for one quote, use it for all quotes by that character
- Add aliases as needed (e.g., "the old man" → "old_man" with alias "stranger")

### Speech Inside Quotes
- Label the actual speaker of the dialogue, not the narrator of the frame
- Example: *"Harry said, 'I'm leaving.'"* → The quote inside is still Harry, not the narrator

### Multi-Paragraph Quotes
- If a quote spans multiple paragraphs, label all segments with the same speaker
- The DialogueSegmenter already tracks multi-paragraph quotes with the `continues_previous` flag
- Your labels should reflect this continuity

### Letters and Diaries
- For first-person epistolary text (letters, diaries), use **NARRATOR_I** for the author/narrator
- If a letter is signed by a character, you can label the author character when they speak in dialogue
- Example: In *Dracula*, Jonathan Harker's diary entries use NARRATOR_I for narration

### Thoughts and Internal Monologue
- Mark as **NOT_SPEECH** (internal monologue), not as a character speaking
- Example: *I wondered what he meant.* → NOT_SPEECH
- If it's explicitly thought-speech (narrator thinking in quotes), use NARRATOR_I

---

## Worked Examples

### Example 1: Dialogue Tag (Third Person)
```
Text: "Hello," said Sherlock Holmes. "I've been expecting you."

Label for [Q1]: sherlock
Label for [Q2]: sherlock

Reasoning: Both quotes are Sherlock speaking. The tag "said Sherlock Holmes" identifies the speaker.
```

### Example 2: Alternating Dialogue (No Tags)
```
Text: "Where were you?" he demanded. "I was at home," she replied.

Label for [Q1]: detective (or "the_man" if unnamed)
Label for [Q2]: suspect (or "the_woman" if unnamed)

Reasoning: Two characters speaking back and forth. Context (pronouns, previous dialogue) helps identify.
```

### Example 3: First-Person Narration
```
Text: I walked to the window. "I can't sleep," I said.

Label for [Q1]: NARRATOR_I

Reasoning: First-person narrator ("I") speaking in their own voice. Not a separate character.
```

### Example 4: Group Speech
```
Text: "We won't stand for this!" they all shouted together.

Label for [Q1]: GROUP

Reasoning: Multiple speakers (the crowd) speaking simultaneously.
```

### Example 5: Scare Quotes
```
Text: The so-called "solution" was nothing of the sort.

Label for [Q1]: NOT_SPEECH

Reasoning: Quoted word for emphasis, not actual speech.
```

### Example 6: Letter (Epistolary)
```
Text: [From Jonathan Harker's diary]
      "I must record what happened today," I wrote in my journal.

Label for [Q1]: NARRATOR_I

Reasoning: First-person narrator writing in a diary. The "I" is the narrator, not a character separate from the narrator.
```

### Example 7: Ambiguous Case
```
Text: "Someone is lying," she said. "I know it."

Label for [Q1]: speaker_1 (the woman)
Label for [Q2]: UNKNOWN

Reasoning: The second quote could be either the woman or someone else. Without context, genuinely undeterminable.
```

---

## Tool Usage

### Interface
The tool shows:
- **Previous paragraph**: Context before the quote
- **Highlighted quote**: The current quote to label with its ID (e.g., [Q3])
- **Next paragraph**: Context after the quote
- **Speaker options**: Numbered list of characters seen so far + special labels

### Commands
- **1-9**: Select speaker from numbered list
- **n**: Add new character with aliases
- **u**: Label as UNKNOWN
- **g**: Label as GROUP
- **x**: Label as NOT_SPEECH
- **m**: Flag a missed or wrong span (extractor error)
- **b**: Undo last label
- **s**: Skip (return later)
- **q**: Save and quit

### Autosave
- The tool saves after every label to `data/gold/<chapter>.json`
- If you quit, you can resume where you stopped
- Use `--relabel 0.15` to re-label a random 15% of already-labeled quotes (measures your consistency)

---

## Style Categories (Automatic)

The tool automatically tags each quote with a style category based on the text:
- **tagged**: Has a dialogue tag ("said NAME" or "NAME said")
- **pronoun-tagged**: Has a pronoun tag ("he said", "she replied")
- **untagged**: No dialogue tag, speaker must be inferred from context
- **alternating**: Two-person rapid dialogue with few tags
- **three-or-more**: Scene with 3+ speakers

You can override these tags manually if needed. The final report shows accuracy by style category.

---

## Quality Checks

### Consistency
- Re-label a random 15% of one chapter at least a day later
- This measures your own consistency (agreement % and Cohen's kappa)
- If you disagree with yourself often, the gold set may be noisy

### Quote Alignment
- The tool uses the production DialogueSegmenter to extract quote spans
- If the extractor misses a quote or invents one, flag it with `m`
- Extractor precision and recall are reported separately from attribution accuracy
- Upstream errors count against the system

### Flags
- Use `m` to flag quotes where the extractor made an error
- These are analyzed separately in the error analysis
- They do not affect attribution accuracy (they're upstream issues)

---

## Progress Tracking

The tool reports per-chapter statistics:
- Total quotes in chapter
- Quotes labeled
- Quotes skipped (to return later)
- Quotes flagged (extractor errors)

Review this summary before marking a chapter as complete.

---

## Common Pitfalls

### Don't Guess
- If you genuinely can't tell who is speaking, use **UNKNOWN**
- Don't choose the most likely option—unknown is a valid label

### Don't Use "Narrator" for First-Person "I"
- Use **NARRATOR_I** specifically for first-person narrators
- Regular third-person narration is automatically labeled as "narrator" by the system

### Don't Ignore Aliases
- If a character is called "Sherlock Holmes" in one place and "Holmes" in another, add both as aliases
- This helps the system recognize that they're the same person

### Don't Confuse Frame Narration with Dialogue
- Example: *Harry wrote in his letter: "I am coming."*
- The quote "I am coming" is Harry speaking (a character), not the narrator
- But the frame "Harry wrote in his letter" is narration

---

## Example Session

```
Previous paragraph:
The room was dark. A figure stood by the window.

Quote [Q1]:
"Who's there?" a voice called out.

Next paragraph:
No answer came from the shadows.

Speaker options:
1. narrator
2. the_figure
3. unknown
4. GROUP
5. NOT_SPEECH
n = new character with aliases
m = flag missed/wrong span
b = undo last label
s = skip
q = save and quit

> n
Enter character name: the_stranger
Enter aliases (comma-separated, or press Enter): voice,shadow
Added: the_stranger (aliases: voice, shadow)

Speaker options:
1. narrator
2. the_stranger
3. unknown
4. GROUP
5. NOT_SPEECH
n = new character with aliases
m = flag missed/wrong span
b = undo last label
s = skip
q = save and quit

> 2
Labeled [Q1] as the_stranger
Saved to data/gold/chapter_03.json
```

---

## Questions?

If you encounter an edge case not covered here, note it in the tool (use `m` to flag) or ask for clarification. The final report will include an error analysis of systematic issues found during labeling.
