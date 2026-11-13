# Register snapshots

These files hold real responses of the public LINDAS SPARQL endpoint `https://lindas.admin.ch/query`, graph `https://lindas.admin.ch/foj/zefix` for the company data. Nothing here is synthetic or edited.

Files: 135. First retrieval (UTC): 2026-11-13T09:33:41+00:00. Last retrieval (UTC): 2026-11-13T09:36:51+00:00.

Each file has the operation, its parameters, the exact SPARQL text, the retrieval time, the measured response time and the response body as sent by the endpoint.

| Operation | Files |
|---|---|
| `company_record` | 23 |
| `exact_name` | 26 |
| `legal_form_label` | 4 |
| `lookup_uid` | 3 |
| `municipality_label` | 22 |
| `scan_legal_names` | 12 |
| `text_search` | 45 |

The responses are public register data. Some of them hold the name and address of a sole proprietor, so this folder only has the responses of a planned set of queries.

`scripts/record_snapshots.py` stores the planned set. Data source license on opendata.swiss: the SPARQL service is listed under terms_open, the dataset metadata under terms_by.
