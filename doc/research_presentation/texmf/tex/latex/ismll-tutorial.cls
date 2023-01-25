\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ismll-tutorial}[2021/06/09 package ismll-tutorial]
\PassOptionsToClass{a4paper, 10pt, oneside}{scrartcl}
\RequirePackage{xparse}
\RequirePackage{xstring}
\RequirePackage{kvoptions}
\RequirePackage{letltxmacro}

%% initialize variables
\title{}
\author{}
\date{}

%% Package Options Processing
%BEGIN_FOLD
\SetupKeyvalOptions{family=tutorial, prefix=tutorial@}
\DeclareBoolOption{solution}
\DeclareStringOption[2cm]{margin}
\@expandtwoargs\@removeelement{margin}\@classoptionslist\@classoptionslist
%% delete all options only intended for local use (else, these options are passed on to all imported packages!!)
% https://tex.stackexchange.com/q/565236
\@expandtwoargs\@removeelement{solution}\@classoptionslist\@classoptionslist

\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessKeyvalOptions*\relax

% starred version necessary due to cat code
% tex.stackexchange.com/questions/243591
\IfEndWith*{\jobname}{publish} {\tutorial@solutionfalse }{}
\IfEndWith*{\jobname}{solution}{\tutorial@solutiontrue  }{}
%END_FOLD




%% Load Packages
%BEGIN_FOLD
\LoadClass{scrartcl}
% use hypertexnames=false to fix "destination with the same identifier"
% https://tex.stackexchange.com/a/282313/119955
\RequirePackage[hypertexnames=false]{hyperref}
\RequirePackage{geometry}
\RequirePackage[headsepline]{scrlayer-scrpage}
\RequirePackage{tikz}
\RequirePackage[most]{tcolorbox}
\RequirePackage{totpages}
\RequirePackage{placeins}
\RequirePackage{version}
\RequirePackage{totpages}
% load math tools for problem* environment
% When loading bold math, delcare hmmax and bmmax before !!!
\DeclareDocumentCommand{\hmmax}{}{0}
%\DeclareDocumentCommand{\bmmax}{}{0}
\RequirePackage{bm}
\RequirePackage{mathtools}
\RequirePackage{relsize}
%END_FOLD

%% Header/Footer + style options
%BEGIN_FOLD
\geometry{a4paper,
          rmargin=\tutorial@margin,
          lmargin=\tutorial@margin,
          bmargin=2cm,
          tmargin=2cm,
          head=24pt}

\RedeclareSectionCommand[%
  % prevent line break after section
  afterskip=-1ex,
  font=\bfseries\sffamily
]{subsection}

\setkomafont{pageheadfoot}{\small}
% https://tex.stackexchange.com/a/21135/119955
\renewcommand{\thempfootnote}{(\roman{mpfootnote})}  % roman footnote in tcolorbox
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}
\renewcommand{\p@subsection}{\thesection}
%END_FOLD

%% Get rid of warning "destination with the same identifier has been already used"
% https://tex.stackexchange.com/a/331766/119955
%\let\thepage\relax
%\pagenumbering{arabic}
%\AtBeginDocument{%
%  \hypersetup{pageanchor=false}%
%  \begin{titlepage}\thispagestyle{empty}\end{titlepage}%
%  \let\thepage\relax
%  \pagenumbering{arabic}
%  \thispagestyle{empty}%
%  \hypersetup{pageanchor=true}%
%}

%% Environments problem/solution 
%\AtBeginEnvironment{section}{\label{ex: \arabic{section}}}
%\apptocmd{\section}{}{}{}



%BEGIN_FOLD
\DeclareBoldMathCommand{\bstar}{\ensuremath{\star}}


\NewDocumentEnvironment{problem}{s s o m}{%begin code
  \IfBooleanT{#1}{%
    \IfBooleanTF{#2}{% two stars
      \renewcommand{\thesection}{\arabic{section}%
        \texorpdfstring{\rlap{\ensuremath{^{\bstar\bstar}}}}{}%
      }%
    }{% one star
      \renewcommand{\thesection}{\arabic{section}%
        \texorpdfstring{\rlap{\ensuremath{^{\bstar}}}}{}%
      }%
    }%
  }%
  \section{\texorpdfstring{#4 \hfill \IfValueT{#3}{(#3 points)}}{}}%
  \label{ex: \arabic{section}}%
  \renewcommand{\thesection}{\arabic{section}}%
}{%end code
}


\NewDocumentEnvironment{subproblem}{s s o}{
  \IfBooleanT{#1}{%
    \IfBooleanTF{#2}{% two stars
      \renewcommand{\thesubsection}{\Alph{subsection}%
        \texorpdfstring{\rlap{\ensuremath{^{\bstar\bstar}}}}{}%
      }%
    }{% one star
      \renewcommand{\thesubsection}{\Alph{subsection}%
        \texorpdfstring{\rlap{\ensuremath{^{\bstar}}}}{}%
      }%
    }%
  }%
  \subsection{\IfValueT{#3}{[#3p]}}%
  \label{ex: \arabic{section}\Alph{subsection}}%
  \renewcommand{\thesubsection}{\Alph{subsection}}%
}{\par}  % adding \par maked wrapfig work properly for some reason.


\iftutorial@solution
\NewDocumentEnvironment{solution}{o}{
  \begin{tcolorbox}[breakable, fonttitle=\bfseries\Large, title={Solution. \IfValueT{#1}{(#1)}}]%
  }{
  \end{tcolorbox}
}
\else
  \excludeversion{solution}
\fi
%END_FOLD



%% Header & Footer
%BEGIN_FOLD
\NewDocumentCommand{\printlogos}{}{
  \begin{tikzpicture}[remember picture, overlay]
    \node[anchor=north, opacity=0.3, inner sep=0cm, outer sep=0pt] at (current page.north){\includegraphics[height=2cm]{icons/UHI-logo-transparent.pdf}};
  \end{tikzpicture}
}


\ihead{\\\@subtitle\ -- \@date}
\chead{\@title \\ \@author \printlogos}
\ohead{\\ \thepage/\ref{TotPages}}

% remove footer
\ifoot{}
\cfoot{}
\ofoot{}
%END_FOLD

\endinput
