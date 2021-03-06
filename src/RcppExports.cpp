// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// CoMM_Binary
List CoMM_Binary(arma::colvec& Y, arma::colvec& Z, arma::mat& X_1, arma::mat& X_2, double constr);
RcppExport SEXP _CommBinary_CoMM_Binary(SEXP YSEXP, SEXP ZSEXP, SEXP X_1SEXP, SEXP X_2SEXP, SEXP constrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X_1(X_1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X_2(X_2SEXP);
    Rcpp::traits::input_parameter< double >::type constr(constrSEXP);
    rcpp_result_gen = Rcpp::wrap(CoMM_Binary(Y, Z, X_1, X_2, constr));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CommBinary_CoMM_Binary", (DL_FUNC) &_CommBinary_CoMM_Binary, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_CommBinary(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
