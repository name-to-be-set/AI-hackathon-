# Day 3 live prompt results

Ran all six prompts through the Streamlit app on 2026-11-12. All six returned a
report and source links with no UI errors or uncaught exceptions. Citation
numbers stayed within each result's source list.

| Prompt | Time | Sources | Report characters | Result |
|---|---:|---:|---:|---|
| Market research — AI note-taking apps | 11.6s | 10 | 7,450 | Passed |
| Technical comparison — LFP vs. NMC batteries | 12.7s | 9 | 3,856 | Passed |
| Recent developments — open-source LLMs | 41.5s | 11 | 5,447 | Passed |
| Policy — generative AI transparency | 40.4s | 12 | 7,302 | Passed |
| Case studies — small-business support agents | 47.6s | 12 | 5,317 | Passed |
| Evidence review — blue light and sleep | 39.2s | 10 | 5,975 | Passed |

## Issues observed

- The later four runs took about 39-48 seconds; report synthesis is the main
  user-visible delay.
- Groq hit a brief token-per-minute limit during testing. The request completed
  using the Gemini backup model.
- No empty reports, invalid citations, or source-link failures were observed.
