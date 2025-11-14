# Documentazione LaTeX - Progetto HeRoN

Questa cartella contiene la documentazione completa del progetto "Adaptive Decision Making NPC in Crafter" utilizzando l'architettura HeRoN.

## Struttura

```
documentazione/
├── main.tex                    # File principale LaTeX
├── bibliography.bib            # Bibliografia
├── capitoli/
│   ├── capitolo1_introduzione.tex
│   ├── capitolo2_architettura.tex
│   ├── capitolo3_crafter.tex
│   ├── capitolo4_metodologia.tex
│   ├── capitolo5_implementazione.tex
│   ├── capitolo6_risultati.tex
│   └── capitolo7_conclusioni.tex
└── immagini/                   # Directory per le immagini
```

## Compilazione

### Prerequisiti

Installare una distribuzione LaTeX completa:
- **Windows**: MiKTeX o TeX Live
- **Linux**: TeX Live (`sudo apt-get install texlive-full`)
- **macOS**: MacTeX

### Compilare il documento

```powershell
# Entrare nella directory documentazione
cd documentazione

# Compilazione completa (include bibliografia)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Oppure usare latexmk (se disponibile)
latexmk -pdf main.tex
```

### Output

Il file compilato sarà: `main.pdf`

## Contenuto dei Capitoli

1. **Introduzione** - Contesto, motivazione e obiettivi del progetto
2. **Architettura HeRoN** - Descrizione dettagliata dei tre componenti (NPC, Helper, Reviewer)
3. **Environment Crafter** - Spazio di stati, azioni, achievement e reward system
4. **Metodologia** - Processo di implementazione fase per fase
5. **Implementazione** - Dettagli tecnici del codice e algoritmi
6. **Risultati** - Analisi quantitativa e confronto con baseline
7. **Conclusioni** - Sintesi, limitazioni e lavori futuri

## Note

- Le immagini (plot, grafici) vanno inserite nella cartella `immagini/`
- I riferimenti bibliografici sono in `bibliography.bib`
- Per aggiungere nuove citazioni, modificare `bibliography.bib` e usare `\cite{key}` nel testo
- Il documento è configurato per lingua italiana

## Personalizzazione

Per modificare gli autori o il titolo, editare le seguenti righe in `main.tex`:

```latex
\title{...}
\author{...}
\date{...}
```
