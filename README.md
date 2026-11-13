# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## What Was Built
- `zefix_agent/`, the Python package of this project.
  - `register_tool.py`, the only code that talks to the register, with prepared queries, timeouts, retries and snapshots.
  - `validation.py`, the input checks and the name comparison.
- `config/agent_config.json`, the endpoint, limits and timeouts of the register tool.
- `snapshots/responses/`, stored real register responses.
- `tests/`, 13 deterministic tests that do not need the network.
- `scripts/record_snapshots.py`, the snapshot recorder.
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

The deterministic tests run without network:
```
python -m unittest discover -s tests
```

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
I follow the LINDAS guidance "How to Use LINDAS Efficiently and Responsibly" where I can. The tool sends the User-Agent `udacity-capstone-project6/0.1`, waits at least 1 second between requests, retries at most 3 attempts with exponential backoff, caches results and does not use regular expressions. The timeouts are 5 seconds to connect and 20 seconds to read.

One point does not fit the guidance well. It asks to avoid queries that scan large parts of the dataset, and the scan over all legal names is such a query. I keep it because the text index misses some legal names, and I limit it to 25 rows.

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```
