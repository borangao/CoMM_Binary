CoMM Binary
===
Collaborative mixed model to dissecting genetic contributions to complex disease by leveraging regulatory information.

Installation 
===========

To install the development version of CoMM , it's easiest to use the 'devtools' package. Note that REMI depends on the 'Rcpp' package, which also requires appropriate setting of Rtools and Xcode for Windows and Mac OS/X, respectively.

```
#install.packages("devtools")
library(devtools)
install_github("borangao/CoMM_Binary")

A<-read.table("Genotype_1.txt") // Genotype for Gene Expression
n1 = dim(A)[1]
q = dim(A)[2]
A<-as.matrix(A)
Atilde<-read.table("Genotype_2.txt") // Genotype for Trait of Interest
n2<-dim(Atilde)[1]
Atilde<-as.matrix(Atilde) 
sigma_g<-sqrt(0.1/dim(A)[2])
sigma_e<-sqrt(0.9)
sigma_e_2<-sqrt(0.1)
w2 = matrix(rep(1,n2),ncol=1)
w1 = matrix(rep(1,n1),ncol=1)
beta_g<-rnorm(dim(A)[2],0,sigma_g) 
error_1<-rnorm(dim(A)[1],0,sigma_e)
Z = A %*%beta_g +error_1
alpha = 0 //Under Null
error_2<-rnorm(dim(Atilde)[1],0,sigma_e_2)
Z_mis = alpha*(Atilde %*%beta_g) +error_2
pr = exp(Z_mis)/(1+exp(Z_mis))
Y = rbinom(length(Z_mis),1,pr)
result<-CoMM_Binary_Testing(Y,Z,A,Atilde)

```

Usage
===========
```
library(CommBinary)
package?CommBinary
```
Development
===========

This package is developed and maintained by Boran Gao (borang@umich.edu).
