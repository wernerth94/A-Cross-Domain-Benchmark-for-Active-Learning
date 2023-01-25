\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ismll-exam}[2021/06/09  package ismll-exam]
\PassOptionsToClass{a4paper, 11pt, twoside, parskip=false}{scrartcl}
\RequirePackage{xparse}
\RequirePackage{xstring}
\RequirePackage{kvoptions}



%% Package Options Processing
%BEGIN_FOLD
\SetupKeyvalOptions{family=exam, prefix=exam@}

\def\exam@optionserror{
  \PackageError{ismll-exam}{Options 'draft', 'print', 'solution' and 'reserve' are mutually exclusive}{}%
}

\def\exam@style{undefined}

\DeclareVoidOption{draft}{%
  \IfEq{\exam@style}{undefined}{\def\exam@style{draft}}{\exam@optionserror}
}

\DeclareVoidOption{reserve}{%
  \IfEq{\exam@style}{undefined}{\def\exam@style{reserve}}{\exam@optionserror}
}

\DeclareVoidOption{print}{%
  \IfEq{\exam@style}{undefined}{\def\exam@style{print}}{\exam@optionserror}
}

\DeclareVoidOption{solution}{%
  \IfEq{\exam@style}{undefined}{\def\exam@style{solution}}{\exam@optionserror}
}

\DeclareBoolOption[true]{extraBpage}
\DeclareComplementaryOption{noextraBpage}{extraBpage}

\DeclareStringOption[0]{reservepages}
\DeclareStringOption[2cm]{margin}


%% delete all options only intended for local use (else, these options are passed on to all imported packages!!)
% https://tex.stackexchange.com/q/565236
\@expandtwoargs\@removeelement{draft}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{reserve}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{print}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{solution}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{extraBpage}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{noextraBpage}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{reservepages}\@classoptionslist\@classoptionslist
\@expandtwoargs\@removeelement{margin}\@classoptionslist\@classoptionslist

% execute user given options
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessKeyvalOptions*\relax
%END_FOLD


%% Secondary options processing
%BEGIN_FOLD
\IfInteger{\exam@reservepages}{
  \ifnum\exam@reservepages<0
    \PackageError{ismll-exam}{Option "reservepages must be non-negative integer}{}
  \fi
}{
  \PackageError{ismll-exam}{Option "reservepages must be non-negative integer}{}
}

% starred version necessary due to cat code of \jobname
% tex.stackexchange.com/questions/243591
% overwrite with \jobname if applicable, fallback: draft
\IfEndWith*{\jobname}   {draft}    {\def\exam@style{draft}}   {}
\IfEndWith*{\jobname}   {print}    {\def\exam@style{print}}   {}
\IfEndWith*{\jobname}   {solution} {\def\exam@style{solution}}{}
\IfEndWith*{\jobname}   {reserve}  {\def\exam@style{reserve}} {}
\IfEq      {\exam@style}{undefined}{\def\exam@style{draft}}   {}
\PackageInfo{ismll-exam}{Using style option "\exam@style"}{}

\newif\ifexam@draft
\newif\ifexam@print
\newif\ifexam@solution
\newif\ifexam@reserve
\IfEq{\exam@style}{draft}   {\exam@drafttrue}   {\exam@draftfalse}
\IfEq{\exam@style}{print}   {\exam@printtrue}   {\exam@printfalse}
\IfEq{\exam@style}{solution}{\exam@solutiontrue}{\exam@solutionfalse}
\IfEq{\exam@style}{reserve} {\exam@reservetrue} {\exam@reservefalse}
%END_FOLD



%% Loading Class and Packages
%BEGIN_FOLD
\LoadClass{scrartcl}
\RequirePackage{geometry}
\RequirePackage{scrlayer-scrpage}
\RequirePackage{subcaption}
\RequirePackage{tikz}
\RequirePackage{pgf}
\RequirePackage{float}
\RequirePackage{totpages}
\RequirePackage[breakable, most]{tcolorbox}
\RequirePackage{version}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage[normalem]{ulem}
\RequirePackage{array, longtable, booktabs, arydshln}  %arraydashline needs to be loaded after these!
\RequirePackage{bm}
\RequirePackage{xparse}
\RequirePackage{xstring}
%END_FOLD



%% Header/Footer + style options
%BEGIN_FOLD
\geometry{a4paper, rmargin=\exam@margin, lmargin=\exam@margin, bmargin=1.5cm, tmargin=2cm}
\DeclareCaptionSubType*[alph]{figure}
\renewcommand{\thesubsection}{\thesection\Alph{subsection}}
\renewcommand{\thesubfigure}{\thefigure\alph{subfigure}}
\setkomafont{pageheadfoot}{\small}

\NewDocumentCommand{\printlogos}{}{
  \begin{tikzpicture}[remember picture, overlay]
    \node[anchor=north, opacity=0.3, inner sep=0cm, outer sep=0pt] at (current page.north){\includegraphics[height=2cm]{icons/UHI-logo-transparent.pdf}};
  \end{tikzpicture}
}

% def. head and foot (co = center odd, le = left even, etc.)
\KOMAoptions{headsepline=true}
\lohead{Name:}
\lehead{\thedate}
\cohead{Matrikel:}
\cehead{\thetitle{} -- \thesubtitle{} \printlogos}
\rohead{\thepage/\ref{TotPages}}
\rehead{\thepage/\ref{TotPages}}
% empty footer
\lofoot{}\lefoot{}\cofoot{}
\cefoot{}\rofoot{}\refoot{}
%END_FOLD



%% Environments problem/solution 
%BEGIN_FOLD
% Automatically insert `\label` with every `\section` and subsection
\let\oldsection\section
\RenewDocumentCommand{\section}{s o m}{%
  \IfBooleanTF{#1}{% starred version -> no label
    \IfValueTF{#2}{% can't use star and short title together
      \PackageError{ismll-tutorial}{either \section*{title}, \section{title} or \section[short title]{title}}{}%
    }{\oldsection*{#3}}
  }{% non-starred version
    \IfNoValueTF{#2}{\oldsection{#3}}{\oldsection[#2]{#3}}
    \label{ex: \arabic{section}}%
  }
}


\let\oldsubsection\subsection
\RenewDocumentCommand{\subsection}{s o m}{%
  \IfBooleanTF{#1}{% starred version -> no label
    \IfValueTF{#2}{% can't use star and short title together
      \PackageError{ismll-tutorial}{either \subsection*{title}, \subsection{title} or \subsection[short title]{title}}{}%
    }{\oldsubsection*{#3}}
  }{% non-starred version
    \IfNoValueTF{#2}{\oldsubsection{#3}}{\oldsubsection[#2]{#3}}
    \label{ex: \arabic{section}\Alph{subsection}}%
  }
}


\NewDocumentCommand{\exam@answerbox}{}{
  % empty box over full page height available
  \begin{tcolorbox}[height fill,  colback=white, colframe=black]
  \end{tcolorbox}
}


\NewDocumentEnvironment{problem}{O{--??--} m}{
  \subsection{\texorpdfstring{#2 \hfill \IfValueT{#1}{(#1 points)}}{}}%
}{% end code:
  \ifexam@print
    \exam@answerbox
    \clearpage
    \ifexam@extraBpage
      \ifnum\value{subsection}=2
        \exam@answerbox
        \clearpage
      \fi
    \fi
  \fi
}


\ifexam@solution
\NewDocumentEnvironment{solution}{o}{
  \begin{tcolorbox}[breakable, fonttitle=\bfseries\Large, title={Solution. \IfValueT{#1}{(#1)}}]%
  }{
  \end{tcolorbox}
}
\else
  \excludeversion{solution}
\fi
%END_FOLD



%% Title page relevant stuff
%BEGIN_FOLD
\let\oldmaketitle\maketitle
\ifexam@print
  \title      {\thetitle}
  \subtitle   {\thesubtitle}
  \author     {\theauthor}
  \date       {\thedate}
  \publishers {\thepublisher}
  \RenewDocumentCommand{\maketitle}{}{
    \begin{titlepage}
      \oldmaketitle
      \printlogo
      \printpoints
      \printdisclaimer
      \thispagestyle{empty}
    \end{titlepage}
  }
\fi
\ifexam@draft
  \RenewDocumentCommand{\maketitle}{}{
    \thispagestyle{empty}
    \printlogo
    \begin{center}
      {\Huge\bfseries\sffamily\thetitle}
      \smallskip
      \\{\bfseries\sffamily\thesubtitle}
%      \smallskip
%      \\{\Huge\bfseries\sffamily\color{red} DRAFT}
      \smallskip
      \\{\Large\thedate}
  \end{center}
  }
\fi
\ifexam@solution
  \RenewDocumentCommand{\maketitle}{}{
    \thispagestyle{empty}
    \printlogo
    \begin{center}
      {\Huge\bfseries\sffamily\thetitle}
      \smallskip
      \\{\bfseries\sffamily\thesubtitle}
      \smallskip
      \\{\Huge\bfseries\sffamily\color{red} SOLUTIONS}
      \smallskip
      \\{\Large\thedate}
  \end{center}
  }
\fi

\NewDocumentCommand{\printlogo}{}{
  \begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north east, opacity=1.0, inner sep=1cm] at (current page.north east){\includegraphics[width=4cm]{icons/UHI-logo-transparent.pdf}};
  \end{tikzpicture}
}

\NewDocumentCommand{\printpoints}{}{
  \begin{table}[H]
  \centering\Large
  \renewcommand{\arraystretch}{2} % temporary increase vertical space
  \begin{tabular}{ll@{\hspace{3cm}} r @{\hspace{1.5cm}} c @{\hspace{1.5cm}} l  r @{\hspace{1cm}} r}
      \bfseries Problem && \bfseries A & \bfseries B & \bfseries C & $\bm\Sigma$&  \\
      \midrule
      \multicolumn{2}{l}{\ref{ex: 1}. \nameref{ex: 1}} &&&&& /10 \\\hdashline
      \multicolumn{2}{l}{\ref{ex: 2}. \nameref{ex: 2}} &&&&& /10 \\\hdashline
      \multicolumn{2}{l}{\ref{ex: 3}. \nameref{ex: 3}} &&&&& /10 \\\hdashline
      \multicolumn{2}{l}{\ref{ex: 4}. \nameref{ex: 4}} &&&&&/10 \\
      \midrule
      First Name: &\multicolumn{4}{l}{\uline{\hfill}} & Bonus: & /04 \\
      Last Name:  &\multicolumn{4}{l}{\uline{\hfill}} & Total: & /40 \\ \cline{6-7}
      Matrikel:   &\multicolumn{4}{l}{\uline{\hfill}} & \multicolumn{2}{|l|}{Grade:} \\ \cline{6-7}
  \end{tabular}
  \end{table}
}

\NewDocumentCommand{\printdisclaimer}{}{%
  \section*{Note:}
\begin{large}
  \begin{itemize}
  \item Time: 120 minutes
  \item Add your name and matrikel on top of every \textbf{odd numbered page}.
  \item No electronic devices besides scientific calculators and clocks are allowed!
        If we catch you using any other devices you automatically failed the exam.
  \item Write with a \textbf{non-erasable pen},do not use red color.
  \item Use the \textbf{reserve pages} at the end if you run out of space. 
        If you still need more, ask staff for rextra pages.
  \item \textbf{Always} explain your reasoning unless asked otherwise.
  \end{itemize}
\end{large}
}
%END_FOLD

%% Reserve pages at end/reserve
%BEGIN_FOLD
\NewDocumentCommand{\exam@reservepage}{}{
  \oldsection*{Reserve Page -- clearly indicate which problem you are working on!}%
  \begin{tcolorbox}[height fill,  colback=white, colframe=black]%
    %
  \end{tcolorbox}
}

% blank page -- no content except header
\NewDocumentCommand{\exam@blankpage}{}{
  \ifexam@print
    \clearpage % clear previous page
    \null      % inserts empty character
    \clearpage % clear the new page
  \fi
}

\IfStrEqCase{\exam@style}{
  % uses `provide` or `declare` based on flag variable
  {draft}{%
    \let\reservepage\relax
    \let\blankpage\relax
  }{solution}{%
    \let\reservepage\relax
    \let\blankpage\relax
  }{print}{%
    \let\reservepage\exam@reservepage
    \let\blankpage\exam@blankpage
  }{reserve}{%
    \let\reservepage\exam@reservepage
    \let\blankpage\exam@blankpage
}}[\PackageError{ismll-exam}{option \exam@style unknown}{}]

% when printing reserve pages terminate early
\ifexam@reserve
\AtBeginDocument{
  \exam@reservepage
  \exam@reservepage
  \end{document}
}
\fi
