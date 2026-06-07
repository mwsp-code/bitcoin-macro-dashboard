# Contributing

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Run the dashboard:

```powershell
streamlit run app.py
```

Run validation:

```powershell
python -m py_compile app.py
pytest -q
```

## Branch Workflow

1. Update `main`: `git switch main && git pull --ff-only`.
2. Create a focused branch: `git switch -c feature/short-description`.
3. Make one logical change at a time.
4. Run compilation and tests.
5. Update `CHANGELOG.md` for user-visible changes.
6. Push the branch and open a pull request into `main`.

Recommended branch prefixes:

- `feature/` for new functionality
- `fix/` for defects
- `model/` for feature or methodology changes
- `ui/` for dashboard presentation
- `docs/` for documentation

## Data And Model Changes

Any change to instruments, feature timing, target construction, training windows,
transaction costs, or optimization must document:

- the exact data source and symbol;
- whether values were available at prediction time;
- the training and evaluation periods;
- out-of-sample performance before and after the change;
- turnover, drawdown, and transaction-cost assumptions.

Do not commit generated cache files. Tests must use synthetic or explicitly
licensed fixtures.

## Pull Request Checklist

- [ ] The dashboard starts without exceptions.
- [ ] `python -m py_compile app.py` passes.
- [ ] `pytest -q` passes.
- [ ] No secrets, local paths, or generated caches are committed.
- [ ] Data-source fallbacks remain clearly labeled.
- [ ] Backtest changes avoid look-ahead leakage.
- [ ] Documentation and changelog are updated.
