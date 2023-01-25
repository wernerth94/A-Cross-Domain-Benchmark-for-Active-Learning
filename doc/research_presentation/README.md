# The ismll-slides document class

## Prerequisites

1. It may be necessary to install the latest version of TeX Live (available at <https://www.tug.org/texlive/>). (for example, some features are not present in TeX Live 2019 distribution)

2. In order to use the template you need to install the ismll-package stack in your `TEXMF` homefolder. You can execute

```bash
echo $TEXMFHOME
```

to check if it already exists. Else, create it in your home directory

```bash
mkdir -p ~/texmf/tex/latex
```

And copy the files from the `texmf` folder from this repository into there.

An easy way to compile both the slides and handout version is to simply execute

```bash
bash makepresentations.sh
```

## Highlights

TODO
