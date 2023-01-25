\NeedsTeXFormat{LaTeX2e}
\RequirePackage{silence}
\WarningFilter{beamerthememetropolis}{You need to compile with XeLaTeX or LuaLaTeX to use the Fira fonts}
\ProvidesClass{ismll-slides}[2021/06/15 package ismll-slides]
\PassOptionsToClass{aspectratio=169, 11pt}{beamer}
\RequirePackage{xparse}
\RequirePackage{xstring}
\RequirePackage{kvoptions}

%% Package Options Processing
%BEGIN_FOLD
\SetupKeyvalOptions{family=slides, prefix=slides@}
%\DeclareBoolOption{handout}
\DeclareStringOption[all]{logo}
\DeclareBoolOption[true]{logotitlepage}
\DeclareBoolOption[false]{invertlogo}
\DeclareBoolOption[true]{pdfpagenumbers}
\DeclareComplementaryOption{nopdfpagenumbers}{pdfpagenumbers}
%% delete all options only intended for local use (else, these options are passed on to all imported packages!!)
% https://tex.stackexchange.com/q/565236
\@expandtwoargs\@removeelement{logo}\@classoptionslist\@classoptionslist
%\@expandtwoargs\@removeelement{handout}\@classoptionslist\@classoptionslist
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessKeyvalOptions*\relax


\newif\ifslides@logoheadline
\newif\ifslides@logoframetitle
\IfEqCase{\slides@logo}{
  {all}       {\slides@logoheadlinefalse  \slides@logoframetitletrue  \slides@logotitlepagetrue }%
  {frametitle}{\slides@logoheadlinefalse  \slides@logoframetitletrue }%
  {headline}  {\slides@logoheadlinetrue   \slides@logoframetitlefalse }%
  {none}      {\slides@logoheadlinefalse  \slides@logoframetitlefalse \slides@logotitlepagefalse}%
}[\PackageError{ismll-slides}{Key=logo only allowed options are [all, title, header, none]}{}]


\ifslides@invertlogo
  \def\slides@logofile{icons/UHI-logo-transparent-inverted.pdf}
\else
  \def\slides@logofile{icons/UHI-logo-transparent.pdf}
\fi
%END_FOLD


%% Load Packages
%BEGIN_FOLD
\LoadClass{beamer}
%% change mode depending on jobname for batch processing
% tex.stackexchange.com/questions/243591
\IfEndWith*{\jobname}   {publish}    {\def\beamer@currentmode{beamer}} {}
\IfEndWith*{\jobname}   {handout}    {\def\beamer@currentmode{handout}}{}

\RequirePackage{etoolbox}
\RequirePackage{bookmark}
\RequirePackage{xmpmulti}
\usepackage[sfdefault]{FiraSans}
\RequirePackage{pdfpages}
\RequirePackage{ismll-packages}
\RequirePackage{ismll-mathoperators}
\RequirePackage{ismll-shortsyms}
\RequirePackage{ismll-style}
\RequirePackage[pangram]{blindtext}
\RequirePackage{tikz}
\RequirePackage{xcolor}
%END_FOLD


%%Bibliography
%BEGIN_FOLD
\RequirePackage[backend=biber, style=numeric]{biblatex}
\AtEveryBibitem{
% Comment those fields one wants to keep. By default, almost everything is shown that is available
% TODO: some of these are not fields, but list. Check biblatex packlage documentation and replace when appropriate. Also complete missing entries in the list.

% Remove INDIVIDUALS
%\clearname{author}
%\clearname{bookauthor}
\clearname{editor}
\clearname{editora}
\clearname{editorb}
\clearname{editorc}
\clearfield{afterword}
\clearfield{annotator}
\clearfield{commentator}
\clearfield{forward}
\clearfield{introduction}
\clearfield{translator}
\clearfield{holder}
% Remove ORGS
\clearfield{institution}
\clearfield{organization}
\clearlist{publisher}
% Remove TITLES
%\clearfield{title}
%\clearfield{indextitle}
%\clearfield{booktitle}
%\clearfield{maintitle}
%\clearfield{journaltitle}
%\clearfield{issuetitle}
%\clearfield{eventtitle}
%\clearfield{reprinttitle}
%\clearfield{series}
% Remove VOLUME & VERSION
\clearfield{volume}
\clearfield{number}
\clearfield{part}
\clearfield{issue}
\clearfield{volumes}
\clearfield{edition}
\clearfield{version}
\clearfield{pubstate}
% Remove PAGES
\clearfield{pages}
\clearfield{pagetotal}
\clearfield{(book)pagination}
% Remove DATES
%\clearfield{date}
\clearfield{eventdate}
\clearfield{urldate}
\clearfield{urlyear}
\clearfield{urlmonth}
\clearfield{urlday}
% Remove LOCATIONS
\clearlist{location}
\clearfield{venue}
% Remove Digital
%\clearfield{url}
\clearfield{doi}
\clearfield{eid}
\clearfield{eprint}
\clearfield{eprinttype}
% Remove TYPES
\clearfield{type}
\clearfield{entrysubtype}
% Remove MISC
\clearfield{addendum}
\clearfield{note}
\clearfield{howpublished}
\clearfield{publisher}
% Remove INTERNATIONAL STANDARDS
\clearfield{isan}
\clearfield{isbn}
\clearfield{ismn}
\clearfield{isrn}
\clearfield{issn}
\clearfield{iswc}
% Remove LABELS
\clearfield{label}
\clearfield{labelyear}
\clearfield{labelmonth}
\clearfield{labelday}
\clearfield{shorthand}
\clearfield{shorthandintro}
}
\renewcommand*{\bibfont}{\scriptsize}
%END_FOLD

%% Patch author field
% https://tex.stackexchange.com/a/10557/119955
%BEGIN_FOLD
\LetLtxMacro{\oldauthor}{\author}
\RenewDocumentCommand{\author}{m}{\oldauthor{\texorpdfstring{#1}{}}}
%END_FOLD


%% Add number for each slide in pdf toc
% https://tex.stackexchange.com/a/66520/119955
\ifslides@pdfpagenumbers
\apptocmd{\beamer@@frametitle}{\only<1>{\bookmark[page=\the\c@page,level=3]{#1}}}%
{\message{** patching of \string\beamer@@frametitle succeeded **}}%
{\message{** patching of \string\beamer@@frametitle failed **}}%
\fi


%% Beamer Theme
%BEGIN_FOLD
% Fix spurios space in metropolis theme
% https://tex.stackexchange.com/a/444420/119955
\AfterEndPreamble{
\def\titlepage{%
  \usebeamertemplate{title page}%<---
}
}

% Create \frametitleheight macro.
% https://tex.stackexchange.com/a/401410/119955
\RequirePackage{printlen}
\AtEndPreamble{
% Execute at End Preamble after theme is loaded
\newlength{\frametitelheight}% <- NEW
\newsavebox{\slides@frametitlebox}% <- NEW
%
\setbeamertemplate{frametitle}{%
  \ifbeamercolorempty[bg]{frametitle}{}{\nointerlineskip}%
  \@tempdima=\textwidth%
  \advance\@tempdima by\beamer@leftmargin%
  \advance\@tempdima by\beamer@rightmargin%
  \sbox{\slides@frametitlebox}{% <- NEW
      \begin{beamercolorbox}[sep=0.3cm,left,wd=\the\@tempdima]{frametitle}
        \usebeamerfont{frametitle}%
        \vbox{}\vskip-1ex%
        \if@tempswa\else\csname beamer@fteleft\endcsname\fi%
        \strut\insertframetitle\strut\par%
        {%
          \ifx\insertframesubtitle\@empty%
          \else%
          {\usebeamerfont{framesubtitle}\usebeamercolor[fg]{framesubtitle}\insertframesubtitle\strut\par}%
          \fi
        }%
        \vskip-1ex%
        \if@tempswa\else\vskip-.3cm\fi% set inside beamercolorbox... evil here...
      \end{beamercolorbox}%
  }% <- NEW
  \usebox{\slides@frametitlebox}% <- NEW
  \settoheight{\frametitelheight}{\usebox{\slides@frametitlebox}}% <- NEW
  \addtolength{\frametitelheight}{\headheight}
}
}
% We need to use the colors once, else they are not initialized.
% https://tex.stackexchange.com/questions/33764/#comment1417533_53604
\AtBeginDocument{
  {\usebeamercolor[bg]{block body}}
  {\usebeamercolor[bg]{block title}}
  {\usebeamercolor[bg]{block title example}}
  {\usebeamercolor[bg]{block title alerted}}
  {\usebeamercolor[bg]{block body alerted}}
  {\usebeamercolor[bg]{block body example}}
}
%END_FOLD


%% Style Options
%BEGIN_FOLD
% make \emph use example text by default
\renewcommand<>{\emph}[1]{%
  {\usebeamercolor[fg]{example text}\onslide#2{}#1}%
}
\captionsetup{font=footnotesize, labelfont={footnotesize, bf}}


\AtEndPreamble{%
% cramped block style
\addtobeamertemplate{block begin}{
%  \vskip-\bigskipamount
  \setlength\abovedisplayskip{.25\baselineskip}
  \setlength\belowdisplayskip{.25\baselineskip}
}{}
\addtobeamertemplate{block example begin}{
%  \vskip-\bigskipamount
  \setlength\abovedisplayskip{.25\baselineskip}
  \setlength\belowdisplayskip{.25\baselineskip}
}{}
\addtobeamertemplate{block alert begin}{
%  \vskip-\bigskipamount
  \setlength\abovedisplayskip{.25\baselineskip}
  \setlength\belowdisplayskip{.25\baselineskip}
}{}
}
%END_FOLD



%% SlimBlock environments
%BEGIN_FOLD
% TODO: to read out /tcb/key values for margins
% https://tex.stackexchange.com/a/565430/119955
\NewDocumentEnvironment{slimblock}{d<> m}{%
\IfValueT{#1}{\onslide<#1>}%
%
%\usebeamertemplate{block}
\begin{tcolorbox}[enhanced,
coltitle=block title.fg,
coltext=block body.fg,
colback=block body.bg,
colbacktitle=block title.bg,
colframe=block title.bg,
title={\bfseries#2},
attach boxed title to top left={yshift=-\tcboxedtitleheight},
boxed title style={size=fbox, sharp corners=all, rounded corners=southeast},
boxed title size=standard,
size=fbox,
sharp corners=northwest,
boxrule=0mm,
frame hidden,
after skip balanced=.25\baselineskip,
before skip balanced=.25\baselineskip,
width=\textwidth+1.5ex,
center,
]
\hspace{\widthof{\bfseries#2}}
}{
\end{tcolorbox}
%
\IfValueT{#1}{\onslide}
}

\NewDocumentEnvironment{slimexampleblock}{d<> m}{%
\IfValueT{#1}{\onslide<#1>}%
%
\begin{tcolorbox}[enhanced, 
coltitle=block title example.fg,
coltext=block body example.fg,
colback=block body example.bg,
colbacktitle=block title example.bg,
colframe=block title example.bg,
title={\bfseries#2},
attach boxed title to top left={yshift=-\tcboxedtitleheight},
boxed title style={size=fbox, sharp corners=all, rounded corners=southeast},
boxed title size=standard,
size=fbox,
sharp corners=northwest,
boxrule=0mm,
frame hidden,
after skip balanced=.25\baselineskip,
before skip balanced=.25\baselineskip,
width=\textwidth+1.5ex,
center,
]
\hspace{\widthof{\bfseries#2}}
}{
\end{tcolorbox}
%
\IfValueT{#1}{\onslide}
}


\NewDocumentEnvironment{slimalertblock}{d<> m}{%
\IfValueT{#1}{\onslide<#1>}%
%
\begin{tcolorbox}[enhanced, 
coltitle=block title alerted.fg,
coltext=block body alerted.fg,
colback=block body alerted.bg,
colbacktitle=block title alerted.bg,
colframe=block title alerted.bg,
title={\bfseries#2},
attach boxed title to top left={yshift=-\tcboxedtitleheight},
boxed title style={size=fbox, sharp corners=all, rounded corners=southeast},
boxed title size=standard,
size=fbox,
sharp corners=northwest,
boxrule=0mm,
frame hidden,
after skip balanced=.25\baselineskip,
before skip balanced=.25\baselineskip,
width=\textwidth+1.5ex,
center,
]
\hspace{\widthof{\bfseries#2}}
}{
\end{tcolorbox}
%
\IfValueT{#1}{\onslide}
}
%END_FOLD


%% Custom Itemization with outline + fix for enumitem
%BEGIN_FOLD
\setbeamertemplate{itemize items}[triangle]
\setbeamertemplate{itemize subitems}[ball]
\setbeamertemplate{itemize subsubitems}[--]


\newlist{MyItemize}{itemize}{3}
\setlist[MyItemize, 1]{label=%
  \setbeamertemplate{itemize item}[triangle]%
  \usebeamercolor[fg]{itemize item}%
  \usebeamertemplate{itemize item}}
\setlist[MyItemize, 2]{label=%
  \setbeamertemplate{itemize subitem}[circle]%
  \usebeamercolor[fg]{itemize subitem}%
  \usebeamertemplate{itemize subitem}}
\setlist[MyItemize, 3]{label=%
  \usebeamercolor[fg]{itemize subsubitem}--}

\renewcommand{\outlinei}{MyItemize}
\renewcommand{\outlineii}{MyItemize}
\renewcommand{\outlineiii}{MyItemize}

% reduce spacing in outline
%\renewcommand{\arraystretch}{1.5}
%\setlength{\belowcaptionskip}{-10pt}
\setlist[MyItemize]{noitemsep, topsep=0pt, 
  label=\usebeamerfont*{itemize item}%
        \usebeamercolor[fg]{itemize item}%
        \usebeamertemplate{itemize item}%
}

% reduce spacing in itemize type environments
%\renewcommand{\arraystretch}{1.5}
%\setlength{\belowcaptionskip}{-10pt}
\setlist[itemize]{noitemsep, topsep=0pt, 
  label=\usebeamerfont*{itemize item}%
        \usebeamercolor[fg]{itemize item}%
        \usebeamertemplate{itemize item}%
}
\setenumerate{label={\arabic*.},noitemsep,topsep=0pt}
%END_FOLD



%% Title Page, Header & Footer
%BEGIN_FOLD
\AtBeginDocument{

\ifslides@logotitlepage
\addtobeamertemplate{title page}{%
\begin{textblock*}{0cm}(0pt,0pt)
\begin{tikzpicture}[remember picture,overlay]
\node[opacity=0.1, anchor=center, inner sep=0, outer sep=0] at (current page.center)
  {\includegraphics[width=\paperwidth, height=\paperheight, keepaspectratio]{icons/UHI-logo-transparent.pdf}};
\end{tikzpicture}%
\end{textblock*}%
}{}
\fi

\ifslides@logoframetitle
\addtobeamertemplate{frametitle}{}{%
\begin{textblock*}{0cm}(0pt,0pt)
\begin{tikzpicture}[remember picture,overlay]
\node[anchor=north east, inner sep=0, outer sep=0, fill=bg] at (current page.north east)
  {\includegraphics[height=\frametitelheight, keepaspectratio]{\slides@logofile}};
%  {\def\svgheight{\frametitelheight}\input{\slides@logofile}};
\end{tikzpicture}%
\end{textblock*}%
}
\fi

\ifslides@logoheadline
\addtobeamertemplate{headline}{}{%
\begin{textblock*}{0cm}(0pt,0pt)
\begin{tikzpicture}[remember picture,overlay]
\node[anchor=north east, inner sep=0, outer sep=0, fill=bg] at (current page.north east) 
  {\includegraphics[height=\headheight, keepaspectratio]{\slides@logofile}};
%  {\def\svgheight{\frametitelheight}\input{\slides@logofile}};
\end{tikzpicture}%
\end{textblock*}%
}%
\fi
}
%END_FOLD



%%  Beamer Aware Commands
%BEGIN_FOLD




% Beamer aware highlight
% https://tex.stackexchange.com/a/75121/119955
\tikzset{%
  remember picture with id/.style={%
    remember picture,
    overlay,
    save picture id=#1,
  },
  save picture id/.code={%
    \edef\pgf@temp{#1}%
    \immediate\write\pgfutil@auxout{%
      \noexpand\savepointas{\pgf@temp}{\pgfpictureid}}%
  }
}

\def\savepointas#1#2{%
  \expandafter\gdef\csname save@pt@#1\endcsname{#2}%
}

\tikzdeclarecoordinatesystem{pic}{%
  \@ifundefined{save@pt@#1}{%
    \pgfpointorigin
  }{%
  \pgfsys@getposition{\csname save@pt@#1\endcsname}\save@orig@pic%
  \pgfsys@getposition{\pgfpictureid}\save@this@pic%
  \pgf@process{\pgfpointorigin\save@this@pic}%
  \pgf@xa=\pgf@x
  \pgf@ya=\pgf@y
  \pgf@process{\pgfpointorigin\save@orig@pic}%
  \advance\pgf@x by -\pgf@xa
  \advance\pgf@y by -\pgf@ya
  }%
}

\newcounter{highlight}
\newcommand{\hlstart}{\tikz[remember picture with id=hlstart\the\value{highlight},baseline=-0.7ex];\hl@start}
\newcommand{\hlend}{\tikz[remember picture with id=hlend\the\value{highlight},baseline=-0.7ex];\hl@end\stepcounter{highlight}}
\newcommand{\fdstart}{\tikz[remember picture with id=hlstart\the\value{highlight},baseline=-0.7ex];\fd@start}
\newcommand{\fdend}{\tikz[remember picture with id=hlend\the\value{highlight},baseline=-0.7ex];\fd@end\stepcounter{highlight}}
\newcommand{\vlstart}{\tikz[remember picture with id=hlstart\the\value{highlight},baseline=-1em];\vl@start}
\newcommand{\vlend}{\tikz[remember picture with id=hlend\the\value{highlight},baseline=0.3ex];\vl@end\stepcounter{highlight}}

\newcommand{\hl@start}[1][]{%
  \hl@draw{highlighter}{#1}{\the\value{highlight}}}

\newcommand{\hl@end}{}

\newcommand{\fd@start}[1][]{%
  \def\fd@args{#1}}

\newcommand{\fd@end}{\def\@tempa{\hl@draw{fader}}\expandafter\@tempa\expandafter{\fd@args}{\the\value{highlight}}\def\fd@args{}}

\newcommand{\vl@start}[1][]{%
  \vl@draw{highlighter}{#1}{\the\value{highlight}}}

\newcommand{\vl@end}{}


\def\hl@sets{%
  \edef\hl@sx{\the\pgf@x}%
  \edef\hl@sy{\the\pgf@y}%
}
\def\hl@sete{%
  \edef\hl@ex{\the\pgf@x}%
  \edef\hl@ey{\the\pgf@y}%
}

\@ifclassloaded{beamer}{

\def\page@node{
  \path (current page.south east)
      ++(-\beamer@rightmargin,\footheight)
  node[
    minimum width=\textwidth,
    minimum height=\textheight,
    anchor=south east
  ] (page) {};
}

}{

  \def\page@node{
    \path (current page.north west)
    ++(\hoffset + 1in + \oddsidemargin + \leftskip,\voffset + 1in + \topmargin + \headheight + \headsep)
    node[
      minimum width=\textwidth - \leftskip - \rightskip,
      minimum height=\textheight,
      anchor=north west
    ] (page) {};
  }

}

\newcommand{\hl@draw}[3]{%
  \begin{tikzpicture}[remember picture,overlay]%
  \page@node
  \tikzset{#2,highlight=#1,every path/.append style={highlight=#1}}%
  \pgfmathsetlengthmacro{\hl@width}{\the\pgflinewidth - 1pt}%
  \coordinate (hlstart) at (pic cs:hlstart#3);
  \coordinate (hlend) at (pic cs:hlend#3);
  \tikz@scan@one@point\hl@sets(pic cs:hlstart#3)
  \tikz@scan@one@point\hl@sete(pic cs:hlend#3)
  \ifdim\hl@sy=\hl@ey\relax
  \draw (hlstart) -- (hlend);
  \else
  \draw (hlstart) -- (hlstart -| page.east);
  \pgfmathsetlengthmacro{\hl@sy}{\hl@sy -\hl@width}%
  \pgfmathsetlengthmacro{\hl@ey}{\hl@ey +\hl@width}%
  \loop\ifdim\hl@sy>\hl@ey\relax
  \draw (0,\hl@sy -| page.west) -- (0,\hl@sy -| page.east);
  \pgfmathsetlengthmacro{\hl@sy}{\hl@sy -\hl@width}%
  \repeat
  \draw (hlend -| page.west) -- (hlend);
  \fi
  \end{tikzpicture}%
}

\newcommand{\vl@draw}[3]{%
  \begin{tikzpicture}[remember picture,overlay]%
  \page@node
  \tikzset{#2,highlight=#1,every path/.append style={highlight=#1}}%
  \pgfmathsetlengthmacro{\hl@width}{\the\pgflinewidth - 1pt}%
  \coordinate (hlstart) at (pic cs:hlstart#3);
  \coordinate (hlend) at (pic cs:hlend#3);
  \tikz@scan@one@point\hl@sets(pic cs:hlstart#3)
  \tikz@scan@one@point\hl@sete(pic cs:hlend#3)
  \ifdim\hl@sx=\hl@ex\relax
  \draw (hlstart) -- (hlend);
  \else
  \draw (hlstart) -- (hlstart |- page.south);
  \pgfmathsetlengthmacro{\hl@sx}{\hl@sx -\hl@width}%
  \pgfmathsetlengthmacro{\hl@ex}{\hl@ex +\hl@width}%
  \loop\ifdim\hl@sx>\hl@ex\relax
  \draw (\hl@sx,0 |- page.north) -- (\hl@sx,0 |- page.south);
  \pgfmathsetlengthmacro{\hl@sx}{\hl@sx -\hl@width}%
  \repeat
  \draw (hlend |- page.north) -- (hlend);
  \fi
  \end{tikzpicture}%
}

\tikzset{%
  highlight/.default=highlighter,
  highlight/.style={
    color=\pgfkeysvalueof{/tikz/#1 colour},
    line width=\pgfkeysvalueof{/tikz/#1 width},
    line cap=\pgfkeysvalueof{/tikz/#1 cap},
    opacity=\pgfkeysvalueof{/tikz/#1 opacity},
  },
  highlighter colour/.initial=yellow,
  highlighter width/.initial=14pt,% <-- Tweak (was 12pt)
  highlighter cap/.initial=butt,
  highlighter opacity/.initial=1,
  fader colour/.initial=gray,
  fader width/.initial=12pt,
  fader cap/.initial=butt,
  fader opacity/.initial=.5,
}



\setbeamercolor{highlighted text}{bg=yellow}
\NewDocumentCommand{\highlight}{d<> m}{%
  \IfValueTF{#1}{%
    {\setbeamercovered{invisible}\alt<#1>{\hlstart#2\hlend}{#2}}%
  }{%
    \hlstart#2\hlend%
  }%
}

\NewDocumentCommand{\vhighlight}{d<> m}{%
  \IfValueTF{#1}{%
    {\setbeamercovered{invisible}\alt<#1>{\vlstart#2\vlend}{#2}}%
  }{%
    \hlstart#2\hlend%
  }%
}

\NewDocumentCommand{\fade}{d<> m}{%
  \IfValueTF{#1}{%
    {\setbeamercovered{invisible}\alt<#1>{\fdstart#2\fdend}{#2}}%
  }{%
    \hlstart#2\hlend%
  }%
}


\LetLtxMacro{\oldhl}{\hl}
\RenewDocumentCommand{\hl}{d<> m}{%
  \IfValueTF{#1}{%
    {\setbeamercovered{invisible}\alt<#1>{\oldhl{#2}}{#2}}%
  }{%
    \oldhl{#2}%
  }%
}%
%END_FOLD


\endinput
