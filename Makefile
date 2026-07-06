.PHONY: format
.PHONY: format-check
.PHONY: checks
.PHONY: tests
.PHONY: notebooks_html

format:
	git ls-files -- "*.py" | xargs isort --profile black
	git ls-files -- "*.py" | xargs black
	# docformatter exit with 3 when reformatting a file, which make xargs fail with 123
	git ls-files -- "*.py" | xargs docformatter --in-place || [ $$? -eq 123 ]
	# rerun docformatter without catching 123 to catch possible other errors
	git ls-files -- "*.py" | xargs docformatter --in-place
	command -v nixfmt >/dev/null 2>&1 && git ls-files -- "*.nix" | xargs nixfmt || true

format-check:
	git ls-files -- "*.py" | xargs isort --profile black --check --diff
	git ls-files -- "*.py" | xargs black --check --diff
	git ls-files -- "*.py" | xargs docformatter --check
	git ls-files -- "*.nix" | xargs nixfmt --check

python_checks:
	ruff check
	basedpyright

tests:
	pytest -v

doc:
	make -C docs clean_quarto_pages && make -C docs quarto_pages
	make -C docs clean && make -C docs html
doc_quick:
	make -C docs clean && make -C docs html
