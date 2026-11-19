# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I built a single agent that identifies Swiss companies in the official Zefix data of the Swiss commercial register. The user gives a company name and optionally a place, an address or a UID. The agent chooses its search tools, compares candidates, keeps the context across questions and returns a structured result with register identifiers, evidence, the retrieval time and the remaining uncertainties.

The agent has four main outcomes, MATCH, AMBIGUOUS, NO_MATCH and TECHNICAL_ERROR, and a fifth one, INVALID_INPUT, for input that is rejected before any search. A deterministic check verifies every decision of the LLM against what the register really returned.

This project connects to my earlier capstone projects, which all work with company data for compliance checks. Projects 3 to 5 named a due diligence agent as a later step. Identifying the right company is the first thing such an agent has to do, since every later check needs to know which company is meant.

## What Was Built
- `zefix_agent/`, the Python package of the agent.
  - `graph.py`, the LangGraph StateGraph with its nodes, conditional edges and memory.
  - `register_tool.py`, the only code that talks to the register, with prepared queries, timeouts, retries and snapshots.
  - `verification.py`, the deterministic checks of the decisions.
  - `validation.py`, the input checks and the name comparison.
  - `schemas.py`, the tool arguments and the structured result.
- `prompts/system_prompt.md`, the persona and the rules of the agent.
- `config/agent_config.json`, the model, limits and timeouts.
- `snapshots/responses/`, stored real register responses.
- `tests/`, 39 deterministic tests that need neither a key nor the network.
- `scripts/record_snapshots.py`, the snapshot recorder.
- `.env.example`, the template for the key and the optional endpoint.
- `requirements.txt`, generated with `pip freeze` in the virtual environment of this project.

## Dataset
Zefix, Central Business Name Index, published by the Federal Commercial Registry Office on opendata.swiss:
https://opendata.swiss/en/dataset/zefix-zentraler-firmenindex

I query it only through the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query` and the graph `https://lindas.admin.ch/foj/zefix`. The dataset description says that it holds daily updated core data of the active legal entities in the commercial register. There is no status field and no date of registration in the data.

I did not use this source in my earlier capstone projects. Project 1 used GLEIF data, Project 2 used a file of register messages from Basel-Landschaft, and Project 5 used company names from the UK register.

No data file has to be downloaded. The stored register responses in `snapshots/responses/` are included. They are public register data, and some of them hold the name and address of a sole proprietor, so I stored only the responses of a planned set of queries.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment. LangGraph 1.x needs Python 3.10 or newer, I used Python 3.13.2. The general project setup page says Python 3.9 or newer, but that is not enough for LangGraph 1.x.
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

The deterministic tests run without a key and without network:
```
python -m unittest discover -s tests
```

## Configuration
These environment variables change the behavior without any change in the code.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none, required | The key for the LLM |
| `OPENAI_BASE_URL` | OpenAI default | The LLM endpoint. The course workspace needs `https://openai.vocareum.com/v1` here, because a Vocareum key does not work on the OpenAI default endpoint. |
| `OPENAI_MODEL` | `gpt-6-luna` | The model name. A different model may need other settings in `config/agent_config.json`. |
| `ZEFIX_MODE` | `snapshot` | Where register answers come from, see below |

The file `.env` from step 4 can set each of these variables. `.env.example` lists the key and `OPENAI_BASE_URL`. The other two variables are only needed to change a default, because the model is set in `config/agent_config.json` and the tests and the recorder set their register mode in the code. If a variable is already set in the shell, the code uses that value and not the one in the file. `.env` must not be submitted.

## Model Settings
I use `gpt-6-luna` through the Chat Completions API with `reasoning_effort` set to `none`. OpenAI allows function calling for this model in Chat Completions only with this value, and it allows `temperature` only when the effort is `none`. The client library would move this model to the Responses API on its own, so `use_responses_api` is set to `false` in the config file. The course demos use `gpt-4o-mini`, which OpenAI will shut down on January 14, 2027, and which it replaces with `gpt-6-luna`. A custom endpoint such as the course workspace may not offer this model.

## Snapshot and Live Mode
By default the register tool answers only from stored real responses, so its answers are repeatable. The environment variable `ZEFIX_MODE` changes this.

| `ZEFIX_MODE` | Behavior |
|---|---|
| `snapshot` | Reads stored responses. A query without a stored response is a controlled error, never an empty result. |
| `live` | Asks the endpoint for each new query, with a cache inside the process. |
| `record` | Uses a stored response if there is one, otherwise asks the endpoint and stores the answer. |

Every snapshot is one JSON file with the operation, the parameters, the exact SPARQL text, the retrieval time, the measured response time and the response word for word. `scripts/record_snapshots.py` stores the planned set again.

Simulated failures for the tests come from a `FaultPlan`. Each affected result carries `simulated: true`.

## Use of the Public Endpoint
I follow the LINDAS guidance "How to Use LINDAS Efficiently and Responsibly" where I can. The tool sends the User-Agent `udacity-capstone-project6/0.1`, waits at least 1 second between requests, makes at most 3 attempts with exponential backoff, caches results and does not use regular expressions. The timeouts are 5 seconds to connect and 20 seconds to read.

One point does not fit the guidance well. It asks to avoid queries that scan large parts of the dataset, and the scan over all legal names is such a query. I keep it because the text index misses some legal names, and I limit it to 25 rows.

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```
