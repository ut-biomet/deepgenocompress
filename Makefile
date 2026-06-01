.PHONY: format
.PHONY: format-check
.PHONY: checks
.PHONY: tests
.PHONY: notebooks_html

format:
	git ls-files -- "*.py" | xargs isort --profile black
	git ls-files -- "*.py" | xargs black
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --clear-output --inplace
	command -v nixfmt >/dev/null 2>&1 && git ls-files -- "*.nix" | xargs nixfmt || true

format-check:
	git ls-files -- "*.py" | xargs isort --profile black --check --diff
	git ls-files -- "*.py" | xargs black --check --diff
	git ls-files -- "*.nix" | xargs nixfmt --check

python_checks:
	git ls-files -- "*.py" | xargs ruff check
	git ls-files -- "*.py" | xargs basedpyright

tests:
	pytest -v

notebooks_html:
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --to notebook --execute --inplace
	git ls-files -- "*.ipynb" | xargs jupyter nbconvert --to html

doc:
	make -C docs clean_quarto_pages && make -C docs quarto_pages
	make -C docs clean && make -C docs html
