% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{CoMM_Binary}
\alias{CoMM_Binary}
\title{CoMM Binary}
\usage{
CoMM_Binary(Y, Z, X_1, X_2, constr)
}
\arguments{
\item{Y}{trait vector.}

\item{Z}{gene expression vector.}

\item{X_1}{normalized genotype (cis-SNPs) matrix for eQTL.}

\item{X_2}{normalized genotype (cis-SNPs) matrix for GWAS.}

\item{constr}{indicator for constraint (alpha = 0 if constr = 1).}
}
\value{
List of model parameters
}
\description{
CoMM Binary to dissecting genetic contributions to complex disease by leveraging regulatory information.
}
\author{
Boran Gao, \email{borang@umich.edu}
}
