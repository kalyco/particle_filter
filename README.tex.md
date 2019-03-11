\documentclass{article}
\usepackage[utf8]{inputenc}

\title{Comp 150 Probabilistic Robotics Homework 2: Particle Filter}
%\author{kcomalli}
\date{\vspace{-2em}}

\usepackage{natbib}
\usepackage{graphicx}
\usepackage{url}
\usepackage{amsmath}
\begin{document}

\maketitle

\section{Implementation}

{\bf Requirements}
\begin{itemize}
  \item Python 3
  \item OpenCV
\end{itemize}

{\bf Summary}\\
A particle filter class (particle\_filter.py) is instantiated that takes a name and an openCv image as an argument (provided automatically in main.py)\\
\begin{itemize}
\item name
\item img
\item rows: img row \#
\item cols: img col \#
\item orig\_img: copy of the unmodified image. Used for redrawing
\item iMap: from the ImgMap class, file img\_map.py. Handles setting the origin to the center of the image and converting back when using opencv
\item dt: cur time step. starts at 0, increases by 1
\item state: Randomly generates x,y coords on the map and draws a circle w a 100 px radius
\item P: set of M particle (p) objects randomly distributed across the image. p has a {\bf state}, (x,y coord), {\bf weight} (1/M initially), {\bf image} (reference image), {\bf histogram} (used to calc cv2.HISTCMP\_CORREL, which updates the weight
\end{itemize}


{\bf Usage}\\
 \$ python setup.py\\
 \$ python main.py\\
  
  \begin{tabular}{ |p{3cm}||p{6cm}|  }
 \hline
 \multicolumn{2}{|c|}{ImageCommands} \\
 \hline
 Commands &  Descriptions  \\
 \hline
 m & Show state reference \\
 p  & Show closest particle reference \\
 Enter  & Move*\\
 \hline
\end{tabular}

*Move:  Generate histograms of particles \-->\ inflate particle size according to histogram correlation \-->\ move x,y of state \-->\ resample particles \-->\ move/distribute particles accordingly

{\bf Probability}\\

Hypothetical state for particle $x^{[m]}_{t}$ where $m \in M$\\\\
$x^{[m]}_{t}$ = particle x[m] at time t\\
$u_{t}$ = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))], where angle = random.uniform(0, 2.0*math.pi)\\
$x_{t}$ = state\\

$x^{[m]}_{t} = p(x_{t} | u_{t}, x^{[m]}_{t-1})$\\\\
$= \dfrac{p(u_{t}|x_{t},x^{[m]}_{t-1})}{p(u_{t}|x^{[m]}_{t-1})}$\\

In our case the measurement does depend on the state


\end{document}