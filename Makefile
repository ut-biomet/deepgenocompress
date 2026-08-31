.PHONY: format
.PHONY: format-check
.PHONY: checks
.PHONY: tests
.PHONY: notebooks_html

format:
	git ls-files -- "*.py" | xargs isort --profile black
	git ls-files -- "*.py" | xargs black
	command -v nixfmt >/dev/null 2>&1 && git ls-files -- "*.nix" | xargs nixfmt || true

format-check:
	git ls-files -- "*.py" | xargs isort --profile black --check --diff
	git ls-files -- "*.py" | xargs black --check --diff
	git ls-files -- "*.nix" | xargs nixfmt --check

python_checks:
	ruff check
	basedpyright

tests:
	pytest -v -m ""

doc:
	make -C docs clean_quarto_pages && make -C docs quarto_pages
	make -C docs clean && make -C docs html SPHINXOPTS="$(SPHINXOPTS)" O="$(O)"
doc_quick:
	make -C docs clean && make -C docs html SPHINXOPTS="$(SPHINXOPTS)" O="$(O)"
