# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I built a single agent that identifies Swiss companies in the official Zefix data of the Swiss commercial register. The user gives a company name and optionally a place, an address or a UID. The agent chooses its search tools, compares candidates, keeps the context across questions and returns a structured result with register identifiers, evidence, the retrieval time and the remaining uncertainties.

The agent has four main outcomes, MATCH, AMBIGUOUS, NO_MATCH and TECHNICAL_ERROR, and a fifth one, INVALID_INPUT, for input that is rejected before any search. A deterministic check verifies every decision of the LLM against what the register really returned.

This project connects to my earlier capstone projects, which all work with company data for compliance checks. Projects 3 to 5 named a due diligence agent as a later step. Identifying the right company is the first thing such an agent has to do, since every later check needs to know which company is meant.

## What Was Built
- `agentic_system.ipynb`, the notebook with design, runs, evaluation and summary.
- `zefix_agent/`, the Python package of the agent. The notebook, the evaluation and the tests all use this code.
  - `graph.py`, the LangGraph StateGraph with its nodes, conditional edges and memory.
  - `register_tool.py`, the only code that talks to the register, with prepared queries, timeouts, retries and snapshots.
  - `verification.py`, the deterministic checks of the decisions.
  - `validation.py`, the input checks and the name comparison.
  - `schemas.py`, the tool arguments and the structured result.
  - `evaluation.py` and `display.py`, the evaluation and the notebook output.
- `prompts/system_prompt.md`, the persona and the rules of the agent.
- `config/agent_config.json`, the model, limits and timeouts.
- `snapshots/responses/`, stored real register responses.
- `evaluation/cases.json`, the 38 evaluation cases with their expected results.
- `evaluation/results/` and `evaluation/run_history.json`, the logs of my real runs.
- `tests/`, 45 deterministic tests that need neither a key nor the network.
- `scripts/record_snapshots.py`, the snapshot recorder.
- `scripts/run_evaluation.py`, the script that runs the evaluation cases with the real LLM.
- `architecture_diagram.png`, the architecture diagram.
- `.env.example`, the template for the key and the optional endpoint.
- `requirements.txt`, generated with `pip freeze` in the virtual environment of this project.

## Dataset
Zefix, Central Business Name Index, published by the Federal Commercial Registry Office on opendata.swiss:
https://opendata.swiss/en/dataset/zefix-zentraler-firmenindex

I query it only through the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query` and the graph `https://lindas.admin.ch/foj/zefix`. The dataset description says that it holds daily updated core data of the active legal entities in the commercial register. In my count on 2026-11-19 the graph had 797,449 organizations. Section 4 of the notebook shows this count. There is no status field and no date of registration in the data, so the agent makes no statement about status.

I did not use this source in my earlier capstone projects. Project 1 used GLEIF data, Project 2 used a file of register messages from Basel-Landschaft, and Project 5 used company names from the UK register.

No data file has to be downloaded. The stored register responses in `snapshots/responses/` are included. They are public register data, and some of them hold the name and address of a sole proprietor, so I stored only the responses that my runs needed.

## Main Results
In the execution of the notebook on 2026-11-20 the agent passed 6 of 6 development cases, 20 of 22 final cases and 10 of 10 holdout cases, with `gpt-6-luna`. The two failures are F3 and F4, the English and the German name of the Swiss Federal Railways. In both cases the agent found the railway and its Cargo company and asked which one I meant, and I had expected a MATCH.

The evaluation script ran the same code a few minutes earlier. All 38 cases ended with the same outcome as in the notebook, and 31 of 38 with the same tool sequence. Two earlier runs of the script are also stored. The run on 2026-11-19 used a prompt that asked for questions in the language of the user. The LLM then asked 13 of its 17 questions in German, so the agent now asks in English. The second run on 2026-11-20 was made before I added four rules to the deterministic code, and it failed H3, where the spelling Mueller for Müller ended in NO_MATCH. No run triggered one of these four rules. In the notebook and the third script run together the LLM proposed MATCH 30 times and the check accepted all 30, and no result of a stored run named another company than expected.

These are hand-picked cases, and in 7 of them more than one outcome counts as correct. So the numbers show behavior in difficult situations and no hit rate for the register. A typing error can still end in NO_MATCH for a company that exists (case F16), while the same request ended with a question about the right company in Run 5 of the notebook. In case F17 an instruction in the name field steered the search, but it did not lead to a MATCH. Section 8 of the notebook explains the runs, and section 10 lists the limitations and risks. All logs are in `evaluation/results/`.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment. LangGraph 1.x needs Python 3.10 or newer, I used Python 3.13.2. The general project setup page says Python 3.9 or newer, but that is not enough for this agent.
   ```
   python3.13 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and write your key for the OpenAI chat API after `OPENAI_API_KEY=`:
   ```
   cp .env.example .env   # on Windows: copy .env.example .env
   ```
   `.env` is listed in `.gitignore`, so the key is never committed, and the code never prints it.
5. Start Jupyter:
   ```
   jupyter lab
   ```
6. Open `agentic_system.ipynb` and run all cells
   (Kernel > Restart Kernel and Run All Cells).

A full run needs internet access to the LLM endpoint. My last full run took about 6 minutes and used 456,366 input and 13,810 output tokens in the evaluation, about $0.05 at the list price. The register is only asked when the LLM chooses a query that has no stored response, and in section 9, which is a live run.

The deterministic tests run without a key and without network:
```
python -m unittest discover -s tests
```

The evaluation script runs the 38 cases with the real LLM and needs the key from step 4. It uses the register in `record` mode, so a query without a stored response goes to the endpoint once. It writes one log per group of cases to `evaluation/results/` and one line per group to `evaluation/run_history.json`:
```
python -m scripts.run_evaluation
```

## Configuration
These environment variables change the behavior without any change in the code.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none, required | The key for the LLM |
| `OPENAI_BASE_URL` | OpenAI default | The LLM endpoint. The course workspace needs `https://openai.vocareum.com/v1` here, because a Vocareum key does not work on the OpenAI default endpoint. |
| `OPENAI_MODEL` | `gpt-6-luna` | The model name. A different model may need other settings in `config/agent_config.json`. |
| `ZEFIX_MODE` | `snapshot` | Where register answers come from, see below |

The file `.env` from step 4 can set each of these variables. `.env.example` lists the key and `OPENAI_BASE_URL`. The other two variables are only needed to change a default, because the model is set in `config/agent_config.json`, the notebook uses `record` unless `ZEFIX_MODE` is set, and the tests, the recorder and the evaluation script set their register mode in the code. If a variable is already set in the shell, the code uses that value and not the one in the file. `.env` must not be submitted.

## Model Settings
I use `gpt-6-luna` through the Chat Completions API with `reasoning_effort` set to `none`. OpenAI allows function calling for this model in Chat Completions only with this value, and it allows `temperature` only when the effort is `none`. The client library would move this model to the Responses API on its own, so `use_responses_api` is set to `false` in the config file. A custom endpoint such as the course workspace may not offer this model. If the LLM call fails, the cell in section 6 of the notebook stops with a message, so a wrong model or endpoint does not end in a run full of TECHNICAL_ERROR results.

## Snapshot and Live Mode
By default the register tool answers only from stored real responses, so the register part of a run is repeatable. The notebook sets `record` unless `ZEFIX_MODE` is already set, because the words the LLM chooses change between runs.

| `ZEFIX_MODE` | Behavior |
|---|---|
| `snapshot` | Reads stored responses. A query without a stored response is a controlled error, never an empty result. |
| `live` | Asks the endpoint for each new query, with a cache inside the process. |
| `record` | Uses a stored response if there is one, otherwise asks the endpoint and stores the answer. |

Every snapshot is one JSON file with the operation, the parameters, the exact SPARQL text, the retrieval time, the measured response time and the response word for word. `scripts/record_snapshots.py` stores the planned set again, and `scripts/run_evaluation.py` and the notebook add the responses of the queries the agent chose itself.

Simulated failures come from a `FaultPlan` in the tests, in the evaluation cases and in Run 5 of the notebook. Each affected result carries `simulated: true`, and the notebook marks these runs.

## Reproducibility and Remaining Dependencies
Stored register responses do not make the agent reproducible on their own. The decisions come from an LLM, so every run needs the key and the LLM endpoint. I set the temperature to 0 and send the seed 42, but this does not guarantee identical answers. The evaluation script and the notebook ran the same code a few minutes apart and agreed in all 38 outcomes and in 31 of 38 tool sequences, while the outcomes of F3, F16 and H3 changed between the three script runs. A rerun can also choose a search word that has no snapshot. In `record` mode the tool then asks the real endpoint and stores the answer, and in `snapshot` mode this shows up as a `snapshot_miss` in the results.

The LLM endpoint itself can fail. On 2026-11-19 three executions of the notebook ended with many server errors from the LLM endpoint, and the agent stopped these cases as TECHNICAL_ERROR. The stored execution is from the next day, when no LLM call failed.

The live register can differ from the snapshots, because the data is updated daily. Of the 215 stored responses, 135 are from 2026-11-13, 61 from 2026-11-19 and 19 from 2026-11-20.

## Use of the Public Endpoint
I follow the LINDAS guidance "How to Use LINDAS Efficiently and Responsibly" where I can. The tool sends the User-Agent `udacity-capstone-project6/0.1`, waits at least 1 second between requests, makes at most 3 attempts with exponential backoff, caches results and does not use regular expressions. The timeouts are 5 seconds to connect and 20 seconds to read.

One point does not fit the guidance well. It asks to avoid queries that scan large parts of the dataset, and the scan over all legal names is such a query. I keep it because the text index misses some legal names, and I limit it to 2 scans per turn and 25 rows. In section 4 of the notebook the 23 complete scans that were stored when the saved execution started took between 5.192 and 6.561 seconds, and the 5 scans that stopped at the row limit took at most 2.071 seconds. These are observations from my runs and no guarantee.

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```
