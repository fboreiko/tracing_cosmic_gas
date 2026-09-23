# paper_notes -- build the project notes PDF.
#
#   make            main.pdf, rerunning latex until refs settle
#   make figures    refresh Figures/ symlinks from ../plots
#   make watch      rebuild on every save
#   make overleaf   self-contained tree with real files, not symlinks
#   make clean      drop aux files, keep the pdf
#   make distclean  drop the pdf too

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all figures watch overleaf clean distclean
all: figures
	$(LATEXMK) main.tex

figures:
	@python3 link_figures.py

watch: figures
	$(LATEXMK) -pvc main.tex

overleaf:
	@python3 link_figures.py --copy
	@echo "Figures/ now holds real files. Zip this directory for Overleaf,"
	@echo "then run 'make figures' to go back to symlinks."

clean:
	latexmk -c main.tex

distclean:
	latexmk -C main.tex
