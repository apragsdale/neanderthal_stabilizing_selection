manuscript.pdf: manuscript.tex manuscript.bib
	pdflatex manuscript.tex
	bibtex manuscript
	pdflatex supplement.tex
	bibtex supplement
	pdflatex supplement.tex
	pdflatex manuscript.tex

manuscript.ps: manuscript.dvi
	dvips manuscript

manuscript.dvi: manuscript.tex manuscript.bib
	latex manuscript.tex
	bibtex manuscript
	latex manuscript.tex
	latex manuscript.tex
clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper:
	rm -f *.ps *.pdf
