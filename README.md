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
and the connecting sentences of the email use the model. Anyone can also paste their own OpenRouter
key in the sidebar; it is used for their browser session only and never saved.

## Put it online

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) straight from this repository.

1. Sign in with GitHub and choose **Create app**, then **Deploy a public app from GitHub**.
2. Repository `leottawa/Dark-Room`, branch `main`, main file path `app/streamlit_app.py`.
3. Open **Advanced settings**, choose Python 3.12, and paste the secrets:

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."   # the demo key every visitor uses; never commit it
   # Optional:
   # BRIGHTROOM_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
   # BRIGHTROOM_DEMO_RUNS_PER_SESSION = 15
   # BRIGHTROOM_DEMO_RUNS_PER_DAY = 300
   ```

4. Deploy. Secrets can be changed later in the app's settings.

The demo key pays for every visitor, so the app protects it: its model is fixed, each visitor
gets a limited number of model runs per session, and the whole app a limited number per day. A
run costs about $0.0002 with the default model. Also set a credit limit on the key itself at
[openrouter.ai/keys](https://openrouter.ai/keys). Visitors who paste their own key have no limit and
can choose any OpenRouter model. Enquiries are sent to OpenRouter to be read, so the sidebar asks
visitors to use fictional data.

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
