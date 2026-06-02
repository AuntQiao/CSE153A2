# CSE153A2 Symbolic Music Generation Recording Script

Target slide narration: about 19:45. Keep total recording under 21:00 if you play excerpts.

## Slide 1: Open with the task and deliverables (0:00-0:40)
Introduce CSE 153/253 Assignment 2 and the two symbolic music generation tasks trained from scratch on Nottingham folk tunes. Point out that names are placeholders. Mention the final artifacts: `workbook.html`, `symbolic_unconditioned.mid`, and `symbolic_conditioned.mid`.

## Slide 2: Map the deck to the grading rubric (0:40-1:35)
Explain the four rubric sections: exploratory analysis, modeling, evaluation, and related work. Say dataset/model slides are shared because both tasks use the same symbolic pipeline, then the deck splits into Task 1 and Task 2 results.

## Slide 3: Summarize the project pipeline (1:35-2:45)
Walk left to right: Nottingham ABC files -> `music21` parsing -> token streams -> train/validation splits -> trigram and GRU models -> MIDI outputs. Emphasize Task 1 predicts melody only, while Task 2 inserts chord tokens.

## Slide 4: Explain dataset scale and shape (2:45-4:00)
Use the table first: 879 training tunes and 154 validation tunes. The conditioned representation has more tokens and a larger vocabulary because chord tokens are included. Use the figures to show mostly short folk tunes and a stable pitch range.

## Slide 5: Explain the token representation (4:00-5:10)
Notes encode MIDI pitch and duration in sixteenth-note units. `CHORD_*` tokens carry harmonic context and `<BAR>` preserves measure boundaries. Mention the concrete source tune example from `morris.abc::X15`.

## Slide 6: Explain why the GRU is the main model (5:10-6:25)
Frame both tasks as next-token prediction. The GRU learns a hidden state over prior events; the trigram baseline only sees the previous two tokens. The architecture is intentionally compact: embedding, two-layer GRU, linear output.

## Slide 7: Describe training behavior (6:25-7:35)
Loss drops quickly and then flattens. This shows learning without requiring a large run. Mention 12 epochs for both models, with the conditioned model's best validation epoch at 10.

## Slide 8: Compare against the baseline (7:35-8:45)
Validation perplexity is the clearest quantitative result. Task 1: GRU 7.424 vs trigram 25.200. Task 2: GRU 6.543 vs trigram 28.991. This supports learned recurrent context over short local counts.

## Slide 9: Introduce Task 1 (8:45-9:40)
Task 1 is symbolic unconditioned generation. At generation time the model receives only `<START>` and samples a melody. A good output should stay in a plausible register, avoid immediate loops, and show pitch/rhythm variety.

## Slide 10: Evaluate Task 1 (9:40-11:10)
The GRU output has 158 notes, 14 unique pitches, and pitch range `[66, 83]`. Repeat rate is low at 0.065 and large leap rate is 0.083. Main limitation: only 4 unique durations vs 8 in the validation reference.

## Slide 11: Walk through Task 1 code (11:10-12:05)
The loop keeps a hidden state and feeds the last sampled token back into the model. `END` is blocked until the minimum note count is reached, then temperature and top-k sampling choose the next token.

## Slide 12: Introduce Task 2 (12:05-13:00)
Task 2 is symbolic conditioned generation. The model learns `p(melody | chords)` because chord tokens appear before measures during training. At generation time, supplied chords control the harmonic context.

## Slide 13: Explain chord data (13:00-14:00)
The chord vocabulary is skewed. G and D are the most common, followed by A, C, Em, and Am. Training has 81 unique chord tokens; validation has 42. Common chords should be easier to condition on than rare ones.

## Slide 14: Evaluate Task 2 (14:00-15:35)
The conditioned GRU sample has 63 notes and 16 chord tokens. Its chord-tone rate is 0.714. The n-gram baseline is close at 0.698, but the GRU has a much lower large leap rate: 0.048 vs 0.145.

## Slide 15: Walk through Task 2 code (15:35-16:35)
The sampler feeds a chord token, then samples melody tokens until `<BAR>`. The evaluator tracks the active chord and checks whether note pitch classes belong to the chord-tone set.

## Slide 16: Discuss related work (16:35-17:45)
Keep this concise. Nottingham supports the symbolic ABC representation. Recurrent-network music work motivates the GRU. Music Transformer is a stronger long-range model, but the GRU is appropriate here because it trains from scratch and is easy to audit.

## Slide 17: State limitations and next steps (17:45-18:45)
Perplexity measures held-out token prediction, not musical beauty. Chord-tone rate is useful but can penalize valid passing tones or suspensions. Future work: phrase-level constraints, listener ratings, stronger conditioning, and longer-range models.

## Slide 18: Close and cue playback (18:45-19:45)
Summarize: tokenization supports both tasks; the GRU beats the trigram baseline; the conditioned model shows measurable harmonic alignment. Then play short excerpts of `symbolic_unconditioned.mid` and `symbolic_conditioned.mid` only if the total video remains under 21 minutes.
