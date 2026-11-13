# Agentic AI Systems Project: Swiss Company Identification Agent

**GitHub repository:** https://github.com/mauroreverberi/agentic_ai_systems
(all commits and the `develop` branch as well as the `main` branch are visible there)

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

## Dependencies
`requirements.txt` was created inside the project's own virtual environment with:
```
pip freeze > requirements.txt
```
