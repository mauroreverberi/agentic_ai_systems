# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I built a single agent that identifies Swiss companies in the official Zefix data of the Swiss commercial register. The user gives a company name or a UID and optionally a place or an address. The agent chooses its search tools, compares candidates, keeps the context across questions and returns a structured result with register identifiers, evidence, the retrieval time and the remaining uncertainties.

The agent ends a request with MATCH, AMBIGUOUS, NO_MATCH or TECHNICAL_ERROR, or with INVALID_INPUT for input that is rejected before any search. Deterministic Python code checks each proposed outcome against what the register really returned.

This project connects to my earlier capstone projects, which all work with company data for compliance checks. Projects 3 to 5 named a due diligence agent as a later step. Identifying the right company is the first thing such an agent has to do, since every later check needs to know which company is meant.

## What Was Built
- A Jupyter notebook (`agentic_system.ipynb`) with the design, the example runs, the evaluation and the summary.
- A written report (`module_summary.pdf`), the Agentic AI System Design Report.
- A Python package (`zefix_agent/`) with the agent graph, the register tools and the checks of the decisions. The graph is a LangGraph `StateGraph`, which I know from the elective module on LangChain and LangGraph of my Agentic AI coursework. The notebook, the evaluation script and the tests all use this code.
- The persona prompt (`prompts/system_prompt.md`), the settings (`config/agent_config.json`) and the architecture diagram (`architecture_diagram.png`).
- The 38 evaluation cases (`evaluation/cases.json`), the logs of my real runs (`evaluation/results/` and `evaluation/run_history.json`) and the stored real register responses (`snapshots/responses/`, described in `snapshots/README.md`).
- 60 deterministic tests that need neither a key nor the network (`tests/`), and two scripts that record register responses and run the evaluation cases (`scripts/`).
- The template for the key (`.env.example`) and a reproducibility file (`requirements.txt`), generated with `pip freeze` inside the project's own virtual environment.

## Dataset
Zefix, Central Business Name Index, published by the Federal Commercial Registry Office on opendata.swiss:
https://opendata.swiss/en/dataset/zefix-zentraler-firmenindex

I query it only through the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query` and the graph `https://lindas.admin.ch/foj/zefix`. The dataset description says that it holds daily updated core data of the active legal entities in the commercial register. In my count on 2026-11-23 the graph had 790,074 organizations. The records have no status field and no registration date of the company, so the agent makes no statement about status.

I did not use this source in my earlier capstone projects. Project 1 used GLEIF data, Project 2 used a file of register messages from Basel-Landschaft, and Project 5 used company names from the UK register.

No data file has to be downloaded. The stored register responses in `snapshots/responses/` are included. They are public register data, and some of them hold the name and address of a sole proprietor, so I stored only the responses of my data probes, of the companies in my evaluation cases and of my runs.

## Main Results
In the execution of the notebook on 2026-11-23 the agent passed 6 of 6 development cases, 22 of 22 final cases and 9 of 10 holdout cases, with `gpt-6-luna`. The failure is H3, where the agent searched only for the spelling Mueller from the request and ended with NO_MATCH, although two entries with the spelling Müller exist.

Before the notebook, the evaluation script ran the same 38 cases three times with the same code and prompt, and the runs passed 37, 35 and 37 cases. F3, F4 and H3 changed their outcome between the four runs, so the behavior is not fully repeatable. No MATCH named another company than I expected.

These are hand-picked cases, so the numbers show behavior in difficult situations and no hit rate for the register. A typing error ended in NO_MATCH for a company that exists (case F16) in all four runs, and this case still counts as a pass. The notebook compares the runs, and the report describes the limitations and risks. The evaluation logs are in `evaluation/results/`. Some cases simulate failures of the register, and each affected tool result in the log carries `simulated: true`.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment with Python 3.13. I used Python 3.13.2, and the pinned packages need at least Python 3.12.
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

A full run needs a key and internet access to the LLM endpoint. My last full run took about 6 minutes and cost about $0.05 at the list price. I used `gpt-6-luna` with my own OpenAI key on the OpenAI default endpoint. I did not test the course workspace, so I cannot say whether it offers this model. The comment in `.env.example` shows how to set another endpoint with `OPENAI_BASE_URL`, and the variable `OPENAI_MODEL` sets another model name, but a different model may need other settings in `config/agent_config.json`. If the LLM call fails, the cell in section 6 of the notebook stops with a message.

The register answers come from stored real responses. The variable `ZEFIX_MODE` sets this behavior, and the notebook uses `record` unless the variable is set:

- `snapshot` reads stored responses only. A query without a stored response is a controlled error, never an empty result.
- `record` uses a stored response if there is one, otherwise it asks the endpoint and stores the answer.
- `live` asks the endpoint for each new query.

So the real register is only asked when the LLM chooses a query that has no stored response, and in section 9 of the notebook, which is a live run. All 215 stored responses are from 2026-11-23, and the live register can differ from them, because the data is updated daily.

The notebook, this README and the report describe my stored execution of 2026-11-23. A new execution overwrites `evaluation/results/notebook_run.json`, and the LLM can take another path than in the stored execution, so its numbers can differ. Without a key you can still read the saved notebook and the logs, and run the deterministic tests:
```
python -m unittest discover -s tests
```

The evaluation script runs the 38 cases with the real LLM and needs the key from step 4:
```
python -m scripts.run_evaluation
```

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
I stay in the domain of company data and compliance checks. This project adds the first step of the due diligence agent that Projects 3 to 5 named, which is finding the right register entry. A later project could link the identified Zefix entry with the GLEIF data of Project 1. The cases with two register entries for one name show why the identity of a company has to be settled before any other check.
