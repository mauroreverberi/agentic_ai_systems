# Register snapshots

These files hold real responses of the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query`, graph `https://lindas.admin.ch/foj/zefix` for the company data. Nothing here is synthetic or edited.

Files: 215. First retrieval (UTC): 2026-11-23T07:20:05+00:00. Last retrieval (UTC): 2026-11-23T10:42:46+00:00. All of them are from 2026-11-23.

Each file has the operation, its parameters, the exact SPARQL text, the retrieval time, the measured response time and the response body as sent by the endpoint.

| Operation | Files |
|---|---|
| `company_record` | 41 |
| `count_companies` | 1 |
| `exact_name` | 35 |
| `legal_form_label` | 7 |
| `lookup_uid` | 4 |
| `municipality_label` | 29 |
| `scan_legal_names` | 27 |
| `text_search` | 71 |

The responses are public register data. Some of them hold the name and address of a sole proprietor, so this folder only has the responses of my data probes, of the companies in my evaluation cases and of my runs.

`scripts/record_snapshots.py` stores the planned set of data probes and the records of all companies that the evaluation cases name, and `scripts/run_evaluation.py` and the notebook add the responses of queries the agent chose itself. The notebook also stores the count of all organizations (`count_companies`). Data source license on opendata.swiss: the SPARQL service is listed under terms_open, the dataset metadata under terms_by.
