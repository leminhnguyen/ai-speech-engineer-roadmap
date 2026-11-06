PYTHON ?= python3
LESSONS := $(patsubst %/,%,$(sort $(wildcard tutorials/*/[0-9][0-9]-*/)))

.PHONY: audit test figures notebooks sota sota-check check

audit:
	$(PYTHON) scripts/audit_tutorials.py

test:
	@for lesson in $(LESSONS); do $(PYTHON) -m unittest discover -s $$lesson/code/tests -v || exit 1; done

figures:
	@for lesson in $(LESSONS); do $(PYTHON) $$lesson/code/generate_figures.py --output-dir $$lesson/assets/generated || exit 1; done

notebooks:
	@for lesson in $(LESSONS); do (cd $$lesson/notebook && $(PYTHON) -m jupyter nbconvert --to notebook --execute --stdout lesson.ipynb > /dev/null) || exit 1; done

sota:
	$(PYTHON) scripts/build_sota.py

sota-check:
	$(PYTHON) scripts/build_sota.py --check

check: audit test sota-check
