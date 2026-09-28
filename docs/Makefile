SPHINXBUILD ?= sphinx-build

.PHONY: clean html linkcheck docs

clean:
	rm -rf build
	rm -rf source/api/generated

html:
	$(SPHINXBUILD) -b html source build/html -E
	@echo
	@echo "Build finished. The HTML pages are in build/html."

linkcheck:
	$(SPHINXBUILD) -b linkcheck source build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

docs: clean html linkcheck
