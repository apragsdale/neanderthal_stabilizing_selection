manuscript.pdf: manuscript.tex manuscript.bib
	pdflatex manuscript.tex
	pdflatex supplement.tex
	bibtex manuscript
	pdflatex supplement.tex
	pdflatex manuscript.tex
	pdflatex supplement.tex
	pdflatex manuscript.tex

clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper:
	rm -f *.ps *.pdf
