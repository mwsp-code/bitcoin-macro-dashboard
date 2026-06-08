# GitHub Workflow

Repository: `https://github.com/mwsp-code/bitcoin-macro-dashboard`

## Daily Iteration

```powershell
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Make and test the change:

```powershell
python -m py_compile app.py
pytest -q
streamlit run app.py
```

Review and commit:

```powershell
git status
git diff
git add app.py README.md CHANGELOG.md
git commit -m "feat: describe the change"
git push -u origin feature/short-description
```

Open a pull request on GitHub, wait for CI, review the diff, then merge into
`main`.

## Commit Style

- `feat:` new behavior
- `fix:` bug correction
- `model:` methodology or feature change
- `data:` source or parser change
- `ui:` presentation change
- `test:` test coverage
- `docs:` documentation
- `chore:` maintenance

## Releases

After a stable group of changes is merged:

```powershell
git switch main
git pull --ff-only
git tag -a v0.2.0 -m "BTC Macro Dashboard v0.2.0"
git push origin v0.2.0
```

Create a matching GitHub Release and copy the relevant entries from
`CHANGELOG.md`.
