# Register snapshots

These files hold real responses of the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query`, graph `https://lindas.admin.ch/foj/zefix` for the company data. Nothing here is synthetic or edited.

Files: 192. First retrieval (UTC): 2026-11-13T09:33:41+00:00. Last retrieval (UTC): 2026-11-19T12:02:31+00:00. Of these, 135 are from 2026-11-13 and 57 from 2026-11-19.

Each file has the operation, its parameters, the exact SPARQL text, the retrieval time, the measured response time and the response body as sent by the endpoint.

| Operation | Files |
|---|---|
| `company_record` | 37 |
| `exact_name` | 35 |
| `legal_form_label` | 7 |
| `lookup_uid` | 4 |
| `municipality_label` | 30 |
| `scan_legal_names` | 19 |
| `text_search` | 60 |

The responses are public register data. Some of them hold the name and address of a sole proprietor, so this folder only has the responses that my runs needed.

`scripts/record_snapshots.py` stores the planned set, and `scripts/run_evaluation.py` adds the responses of queries the agent chose itself. Data source license on opendata.swiss: the SPARQL service is listed under terms_open, the dataset metadata under terms_by.
