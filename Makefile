.PHONY: help format check test clean


help:
	@echo ""
	@echo "Usage: make <command>"
	@echo ""
	@echo "Available commands:"
	@echo "  - help     show help message"
	@echo "  - format   run formatting tools"
	@echo "  - lint    run python linting tools"
	@echo "  - clean    remove temp python directories and their contents"
	@echo ""


format:
	black main


lint:
	black --check main
	ruff check main
	flake8 --statistics main
	pylint -rn -sn --rcfile=.pylintrc main
	mypy --strict --namespace-packages --explicit-package-bases main


clean:
	@echo "Cleaning up..."
	@find . -type d -name __pycache__ -exec rm -r {} \+
	@find . -type d -name .mypy_cache -exec rm -r {} \+
	@find . -type d -name .pytest_cache -exec rm -r {} \+
	@find . -type d -name .ruff_cache -exec rm -r {} \+
	@find . -name .coverage -exec rm -r {} \+
	@echo "Done!"
