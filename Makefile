.PHONY: format
.PHONY: checks
.PHONY: notebooks_html

format:
	git ls-files -- "*.py" | xargs isort --profile black
	git ls-files -- "*.py" | xargs black
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --clear-output --inplace
	command -v nixfmt >/dev/null 2>&1 && git ls-files -- "*.nix" | xargs nixfmt || true

check:
	ruff check .
	basedpywright .

notebooks_html:
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --to notebook --execute --inplace
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --to html
