# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I built a single agent that identifies Swiss companies in the official Zefix data of the Swiss commercial register. The user gives a company name or a UID and optionally a place or an address. The agent chooses its search tools, compares candidates, keeps the context across questions and returns a structured result with register identifiers, evidence, the retrieval time and the remaining uncertainties.

The agent has four main outcomes, MATCH, AMBIGUOUS, NO_MATCH and TECHNICAL_ERROR, and a fifth one, INVALID_INPUT, for input that is rejected before any search. A deterministic check compares every outcome that the LLM proposes with what the register really returned.

This project connects to my earlier capstone projects, which all work with company data for compliance checks. Projects 3 to 5 named a due diligence agent as a later step. Identifying the right company is the first thing such an agent has to do, since every later check needs to know which company is meant.

## What Was Built
- A Jupyter notebook (`agentic_system.ipynb`) with the design, the example runs, the evaluation and the summary.
- A written report (`module_summary.pdf`), the Agentic AI System Design Report.
- The Python package of the agent (`zefix_agent/`). The notebook, the evaluation and the tests all use this code.
  - `graph.py`, the LangGraph StateGraph with its nodes, conditional edges and memory.
  - `register_tool.py`, the only code that talks to the register, with prepared queries, timeouts, retries and snapshots.
  - `verification.py`, the deterministic checks of the decisions.
  - `validation.py`, the input checks and the name comparison.
  - `schemas.py`, the tool arguments and the structured result.
  - `evaluation.py` and `display.py`, the evaluation and the notebook output.
- The persona and the rules of the agent (`prompts/system_prompt.md`).
- The model, the limits and the timeouts (`config/agent_config.json`).
- Stored real register responses (`snapshots/responses/`).
- The 38 evaluation cases with their expected results (`evaluation/cases.json`).
- The logs of my real runs (`evaluation/results/` and `evaluation/run_history.json`).
- 48 deterministic tests that need neither a key nor the network (`tests/`).
- The snapshot recorder (`scripts/record_snapshots.py`).
- The script that runs the evaluation cases with the real LLM (`scripts/run_evaluation.py`).
- The architecture diagram (`architecture_diagram.png`).
- The template for the key and the optional endpoint (`.env.example`).
- A reproducibility file (`requirements.txt`), generated with `pip freeze` inside the project's own virtual environment.

## Dataset
Zefix, Central Business Name Index, published by the Federal Commercial Registry Office on opendata.swiss:
https://opendata.swiss/en/dataset/zefix-zentraler-firmenindex

I query it only through the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query` and the graph `https://lindas.admin.ch/foj/zefix`. The dataset description says that it holds daily updated core data of the active legal entities in the commercial register. In my count on 2026-11-19 the graph had 797,449 organizations. Section 4 of the notebook shows this count. There is no status field and no date of registration in the data, so the agent makes no statement about status.

I did not use this source in my earlier capstone projects. Project 1 used GLEIF data, Project 2 used a file of register messages from Basel-Landschaft, and Project 5 used company names from the UK register.

No data file has to be downloaded. The stored register responses in `snapshots/responses/` are included. They are public register data, and some of them hold the name and address of a sole proprietor, so I stored only the responses of my data probes and of my runs.

## Main Results
In the execution of the notebook on 2026-11-20 the agent passed 6 of 6 development cases, 20 of 22 final cases and 10 of 10 holdout cases, with `gpt-6-luna`. The two failures are F3 and F4, the English and the German name of the Swiss Federal Railways. In both cases the agent found the railway and its Cargo company and asked which one I meant, and I had expected a MATCH.

The evaluation script had run the same 38 cases three times before that. Its third run used the same agent code a few minutes earlier. All 38 cases ended with the same outcome as in the notebook, and 31 of 38 with the same tool sequence. In the notebook and this script run together the LLM proposed MATCH 30 times and the check accepted all 30, and no result of a stored run named another company than expected. Section 8 of the notebook describes the two earlier script runs and what I changed between them.

After these runs I made two rules of the check for NO_MATCH stricter. A failed fuzzy search no longer counts as a search, and a UID that the register holds cannot end as NO_MATCH. One of the tests reads the stored logs and shows that neither rule would have changed a stored result.

These are hand-picked cases, and in 7 of them more than one outcome counts as correct. So the numbers show behavior in difficult situations and no hit rate for the register. A typing error can still end in NO_MATCH for a company that exists (case F16), while the same request ended as AMBIGUOUS with the right company as the only candidate in Run 5 of the notebook. In case F17 an instruction in the name field steered the search, but it did not lead to a MATCH. Section 10 of the notebook lists the limitations and risks. All logs are in `evaluation/results/`.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment with Python 3.13. I used Python 3.13.2, and the pinned packages need at least Python 3.12. The general project setup page says Python 3.9 or newer, but that is not enough for these package versions.
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

The notebook, this README and the report describe my stored execution of 2026-11-20. A new execution overwrites `evaluation/results/notebook_run.json`, and the LLM can take another path than in the stored execution, so its numbers can differ.

The deterministic tests run without a key and without network:
```
python -m unittest discover -s tests
```

The evaluation script runs the 38 cases with the real LLM and needs the key from step 4. It uses the register in `record` mode, so a query without a stored response is sent to the endpoint. It writes one log per group of cases to `evaluation/results/` and one line per group to `evaluation/run_history.json`:
```
python -m scripts.run_evaluation
```

## Configuration and Register Modes
These environment variables change the behavior without any change in the code. The file `.env` from step 4 can set each of them, and a value that is already set in the shell wins.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none, required | The key for the LLM |
| `OPENAI_BASE_URL` | OpenAI default | The LLM endpoint. The course workspace needs `https://openai.vocareum.com/v1` here, because a Vocareum key does not work on the OpenAI default endpoint. |
| `OPENAI_MODEL` | `gpt-6-luna` | The model name. A different model may need other settings in `config/agent_config.json`. |
| `ZEFIX_MODE` | `snapshot` | Where register answers come from, see below. The notebook uses `record` unless this variable is set. |

I use `gpt-6-luna` through the Chat Completions API with `reasoning_effort` set to `none`, because OpenAI allows function calling for this model in Chat Completions only with this value. All my stored runs used the OpenAI default endpoint and none used the course workspace, so I cannot say whether the workspace offers this model. If the LLM call fails, the cell in section 6 of the notebook stops with a message, so a wrong model or endpoint does not end in a run full of TECHNICAL_ERROR results.

| `ZEFIX_MODE` | Behavior |
|---|---|
| `snapshot` | Reads stored responses. A query without a stored response is a controlled error, never an empty result. |
| `live` | Asks the endpoint for each new query, with a cache inside the process. |
| `record` | Uses a stored response if there is one, otherwise asks the endpoint and stores the answer. |

`snapshots/README.md` describes the stored files. `scripts/record_snapshots.py` fills missing responses of the planned set, and `scripts/run_evaluation.py` and the notebook add the responses of the queries the agent chose itself. Some tests, some evaluation cases and Run 5 of the notebook simulate failures of the register. Each affected tool result in the log carries `simulated: true`.

The register tool sends the User-Agent `udacity-capstone-project6/0.1`, waits at least 1 second between its requests, makes at most 3 attempts with exponential backoff and caches results. The scan over all legal names reads a large part of the dataset, which the LINDAS guidance asks to avoid, so I limit it to 2 scans per turn and 25 rows. The report explains this choice.

## Reproducibility
Stored register responses do not make the agent reproducible on their own. The decisions come from an LLM, so every run needs the key and the LLM endpoint. I set the temperature to 0 and send the seed 42, but this does not guarantee identical answers, and the outcomes of F3, F16 and H3 changed between the three script runs. A rerun can also choose a search word that has no snapshot. In `record` mode the tool then asks the real endpoint and stores the answer, and in `snapshot` mode this shows up as a `snapshot_miss` in the results.

The live register can differ from the snapshots, because the data is updated daily. Of the 215 stored responses, 135 are from 2026-11-13, 61 from 2026-11-19 and 19 from 2026-11-20.

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
I stay in the domain of company data and compliance checks. This project adds the first step of the due diligence agent that Projects 3 to 5 named, which is finding the right register entry. A later project could link the identified Zefix entry with the GLEIF data of Project 1. The cases with two register entries for one name show why the identity of a company has to be settled before any other check.
