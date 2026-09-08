PYTHON ?= python

.PHONY: verify test reproduce figures mock-smoke manifest

verify:
	$(PYTHON) tools/verify_release.py
	$(PYTHON) tools/verify_generators.py
	$(PYTHON) -m unittest discover -s tests -v

test:
	$(PYTHON) -m unittest discover -s tests -v

reproduce:
	$(PYTHON) tools/reproduce.py

figures:
	$(PYTHON) tools/render_figures.py

mock-smoke:
	$(PYTHON) tools/mock_inference_smoke.py

manifest:
	$(PYTHON) tools/build_manifest.py

