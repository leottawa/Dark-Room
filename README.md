# BrightRoom proposal desk

Turns a client enquiry into a structured brief and a draft proposal, or into the clarification
needed before a proposal can be made. HEC Paris MSc DSAIB, Case 02. The company, catalogue and
enquiries are fictional teaching material.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and an OpenRouter API key.

```bash
uv sync
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. The model defaults to
`mistralai/mistral-small-3.2-24b-instruct` (Mistral Small 3.2) and can be changed with `BRIGHTROOM_MODEL`.

```bash
uv run streamlit run app/streamlit_app.py
```

Without a key the app still works from a brief entered by hand: only the reading of the enquiry
and the connecting sentences of the email use the model.

## Test it

```bash
uv run pytest
```

Runs without a key: the pricing engine against the case rules, the test cases against their
answer key, and the model-facing steps against a scripted client.

```bash
uv run python -m brightroom.evaluate --arms oracle pipeline naive engineered --repeats 3
```

Runs every test case through each arm and writes the raw replies, scores, latency and cost to
`eval/results/`.

## Layout

| Path | Contents |
|---|---|
| `data/catalogue.json` | The approved catalogue and policies |
| `src/brightroom/extract.py` | Step 1: the model reads the enquiry; code audits its quotes |
| `src/brightroom/engine.py` | Steps 2-3: routing, pricing, verdict, alternatives (no model) |
| `src/brightroom/draft.py` | Step 4: the email, assembled by code around three model passages |
| `src/brightroom/verify.py` | Step 5: checks on the passages and the assembled draft |
| `src/brightroom/pipeline.py` | The fixed route through steps 1-5, with timing and cost |
| `app/streamlit_app.py` | Step 6: the review and approval screen |
| `src/brightroom/baselines.py` | Single-prompt baselines and the grader for their replies |
| `src/brightroom/scoring.py`, `evaluate.py` | Test cases, checks and the evaluation runner |
| `eval/cases/dev.json` | Development test cases with expected outcomes |
