# ScaleWise — Bioprocess Scale-Up AI App

## Included features

1. User input panel with options/ranges derived from the supplied workbook.
2. 5A physics/scale-up engine.
3. 5B target-scale interpolation.
4. 5C biological outcome prediction:
   - VCD
   - viability
   - growth rate
   - glucose
   - lactate
5. Process Risk Scorecard.
6. Scale-up difficulty/failure-risk warning:
   - warning level
   - risk score
   - evidence/coverage confidence %
   - suggested action
   - recommended validation
7. Interactive AI Copilot using the OpenAI Responses API.

## Important limitation

The supplied workbook contains `failure_event = 0` for all 15,000 rows.
There are no positive failure examples and therefore a supervised failure
classifier cannot be trained honestly.

The app instead calculates an auditable scale-up risk score from:

- operating-envelope excursions,
- scale interpolation/extrapolation,
- multivariate anomaly detection,
- engineering/physics variables.

The displayed confidence is **evidence/coverage confidence**, not a probability
that the batch will fail.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep these files in the same directory:

- `app.py`
- `Final Data for Hackathon.xlsx`
- `requirements.txt`

## Enable the live AI Copilot

Set an OpenAI API key as an environment variable:

Linux/macOS:
```bash
export OPENAI_API_KEY="your_key_here"
```

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Optional:
```bash
export OPENAI_MODEL="gpt-5.6-luna"
```

For Streamlit Cloud, put the API key in the app's Secrets rather than
hard-coding it into the source code.

## Demo target

The sidebar includes 50 L as an interpolated demonstration scale because
50 L is between the observed 10 L and 100 L scales in the supplied dataset.
