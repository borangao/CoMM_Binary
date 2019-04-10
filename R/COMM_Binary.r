#' @author Boran Gao, \email{borang@umich.edu}
#' @title
#' CoMM Binary
#' @description
#' CoMM to dissecting genetic contributions to complex traits by leveraging regulatory information.
#'
#' @param Y  trait vector.
#' @param Z  gene expression vector.
#' @param X_1  normalized genotype (cis-SNPs) matrix for eQTL.
#' @param X_2  normalized genotype (cis-SNPs) matrix for GWAS.
#' @return List of model parameters

CoMM_Binary_Testing<-function(Y,Z,X_1,X_2){
out_null<-CoMM_Binary(Y,Z,X_1,X_2,1)
out_alternative<-CoMM_Binary(Y,Z,X_1,X_2,0)
chisq_stat<-2*(out_alternative$LRT_Dev-out_null$LRT_Dev)
p_value<-pchisq(chisq_stat, df=1, lower.tail=FALSE)
   return( list(
        sigma_e_square = out_alternative$sigma_e_square,
        sigma_g_square = out_alternative$sigma_g_square,
        alpha = out_alternative$alpha,
        p_value = p_value ) )
}
