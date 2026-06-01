# Documentation Build

This directory contains the source files for the documentation.

## Build Process

The documentation is built using [**Sphinx**](https://www.sphinx-doc.org/en/master/). To
build the documentation, run the following command from the project root:

```bash
make doc
```

This command performs the following steps:

1. **Quarto Rendering**: Renders Quarto source files (`.qmd`) to Markdown (`.md`).
1. **Sphinx Build**: Generate the HTML documentation, including both the documentation
   pages and API reference sections generated from the package's docstrings.

## Quarto Pages

We use [Quarto](https://quarto.org/) to create some documentation pages like the examples.

- **Source files**: Located in `docs/quarto_pages/` (e.g., `example.qmd`).
- **Rendered output**: During the build process, Quarto executes the Python code in these
  files, incorporates the code's results into the document, and renders the final output
  as Markdown in the docs/quarto_pages/outputs/ directory.
- **Integration**: The rendered Markdown files are included in the Sphinx documentation by
  referencing them in `docs/index.rst`.

To add a new Quarto page:

1. Create a `.qmd` file in `docs/quarto_pages/`.
1. Update `docs/Makefile` to include the new file in the `quarto_pages` target if it's not
   automatically handled.
1. Add the rendered output path (e.g., `quarto_pages/outputs/your_page`) to the `toctree`
   in `docs/index.rst`.
