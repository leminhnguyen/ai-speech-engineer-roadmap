PYTHON ?= python3
LESSONS := curriculum/01-audio-signal-processing/01-sound-and-waveforms curriculum/01-audio-signal-processing/02-digital-audio

.PHONY: audit test figures notebooks

audit:
	$(PYTHON) scripts/audit_curriculum.py

test:
	@for lesson in $(LESSONS); do $(PYTHON) -m unittest discover -s $$lesson/code/tests -v; done

figures:
	@for lesson in $(LESSONS); do $(PYTHON) $$lesson/code/generate_figures.py --output-dir $$lesson/assets/generated; done

notebooks:
	@for lesson in $(LESSONS); do (cd $$lesson/notebook && $(PYTHON) -m jupyter nbconvert --to notebook --execute --stdout lesson.ipynb > /dev/null); done
