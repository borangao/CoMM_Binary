#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;
colvec logisticGradientLLKc(colvec Y, mat X, colvec beta0) {
  int n = Y.size();
  int m = beta0.size();
  if ( ( size(X)(0) != n ) || ( size(X)(1) != m ) )
    stop("Mismatching dimensions of input data");   // sanity check routine
  
  colvec xtbeta0 = X * beta0 ;
  colvec residual = Y - exp(xtbeta0)/(1+exp(xtbeta0));
  colvec gr(m+1);
  gr.zeros();
  for(int j=0;j<m;j++){
    gr(j) += sum(residual % X.col(j))/n;
  }
  double llk = sum(xtbeta0 % Y - log(exp(xtbeta0)+1))/n;
  gr[m] = llk;
  //  ouble sum_log_exp = 0;
  //  double yAtx = 0;
  //  for(int i=0 ;i < n; i++){
  //    sum_log_exp +=  log(exp(Atx[i])+1);
  //    yAtx += Atx[i]*y[i];
  //  }
  //  double sum = (yAtx - sum_log_exp)/n;
  return gr;
}
double logisticGradientDescent(colvec Y, mat X, colvec beta0, double gamma = 0.1,int max_iter = 1000 , double tol = 1e-5 ) {
  int n = Y.size();
  int m = beta0.size();
  if ( size(X)(0) != n ) stop("Number of rows in X and length of y do not match"); // sanity check 
  if ( size(X)(1) != m ) stop("Number of columns in X and length of beta do not match");
  colvec GradientLLKc = logisticGradientLLKc(Y, X, beta0);
  double llk = GradientLLKc(m); // evaluate initial gradient double prevLoss = loss; // store initial loss
  double prevllk =llk;
  int iter, j;
  colvec xt = beta0;
  for(iter = 1; iter< max_iter; iter++){
    for(j=0; j<m; j++){
      xt[j] +=gamma * GradientLLKc[j];
    }
    GradientLLKc = logisticGradientLLKc(Y, X, xt);
    llk = GradientLLKc[m];
    if(llk>prevllk){
      if((llk-prevllk)/fabs(prevllk)<tol){
        break;
      }else{
        gamma *=1.1;
      }
    }else{
      gamma /=2;
    }
    prevllk = llk;
  }
  colvec gr_out(m);
  for(int i=0;i<m;i++){
    gr_out(i) = GradientLLKc(i);
  }
  double alpha_out = xt(0);
  return(alpha_out); 
}



vec lmm_pxem_ptr(const arma::vec y, const arma::mat x, const int& maxIter,
			  double& sigma2y, double& sigma2beta, double& loglik_max,
			  int& iteration, arma::mat& Sigb, arma::vec& mub){
	if (y.n_elem != x.n_rows){
		perror("The dimensions in outcome and covariates (x) are not matched");
	}

	sigma2y = var(y), sigma2beta = sigma2y;

    int n = y.n_elem, p = x.n_cols;


	if (p != mub.n_elem){
		perror("The dimensions in covariates are not matched in mub");
	}

	if (p != Sigb.n_cols){
		perror("The dimensions in covariates are not matched in Sigb");
	}

	mat xtx = x.t()*x;
	vec xty = x.t()*y;
	double gam, gam2;  // parameter expansion

	vec  dd;
	mat uu;

	eig_sym(dd, uu, xtx);


	vec dd2;
	mat uu2;

	mat tmp = x*x.t();

	eig_sym(dd2, uu2, tmp);

	vec Cm(p), Cs(p), Cm2(p);
	vec loglik(maxIter);
	vec utxty(p), utxty2(p), utxty2Cm(p), utxty2Cm2(p);
	vec yt(n), yt2(n);

	// evaluate loglik at initial values
	vec uty = uu2.t()*(y);

	vec tmpy = uty / sqrt(dd2 * sigma2beta + sigma2y);
	vec tmpy2 = pow(tmpy,2);
	loglik(0) = -0.5 * sum(log(dd2 * sigma2beta + sigma2y)) - 0.5 * sum(tmpy2);

	//cout << "initial :sigma2beta: " << sigma2beta <<"; sigma2y: " << sigma2y << "; loglik: " << loglik(0) << endl;
	int Iteration = 1;
	for (int iter = 2; iter <= maxIter; iter ++ ) {
		// E-step
		Cm = sigma2y / sigma2beta +  dd;
		Cs = 1 / sigma2beta +  dd / sigma2y;
		Cm2 = pow(Cm , 2);
		// M-step
		utxty = uu.t() * (xty); // p-by-1
		utxty2 = pow(utxty, 2);

		utxty2Cm = utxty % utxty / Cm;
		utxty2Cm2 = utxty2Cm/Cm;

		gam = sum(utxty2Cm) / ( sum(dd % utxty2Cm2) + sum(dd / Cs));
		gam2 = pow(gam , 2);
		//cout << " iter: " << iter << "; 1/Cs: " << sum(1 / Cs) << ";utxty2Cm2: " << sum(utxty2Cm2) <<"; gam: " << sum(gam) <<  "; utxty2Cm: " <<	sum(utxty2Cm) <<endl;

		//sigma2beta = ( sum(utxty.t() * diagmat(1 / Cm2) * utxty) + sum(1 / Cs)) / p;
		sigma2beta = ( sum(utxty2Cm2) + sum(1 / Cs)) / p;

		yt = y;
		yt2 = pow(yt , 2);
		sigma2y = (sum(yt2) - 2 * sum(utxty2Cm) * gam + sum(utxty2Cm2 % dd) * gam2 + gam2 * sum(dd / Cs)) / n;


		//reduction and reset
		sigma2beta = gam2 * sigma2beta;

		//evaluate loglik
		uty = uu2.t()*(y);
		tmpy = uty / sqrt(dd2 * sigma2beta + sigma2y);
		tmpy2 = pow(tmpy,2);
		loglik(iter - 1) = - 0.5 * sum(log(dd2 * sigma2beta + sigma2y)) - 0.5 * sum(tmpy2);

		//cout << "sigma2beta: " << sigma2beta <<"; sigma2y: " << sigma2y << "; beta0: " << beta0 << "; loglik: " <<	loglik(iter - 1) << "; sum(tmpy2): " << sum(tmpy2) <<endl;
		if ( loglik(iter - 1) - loglik(iter - 2) < 0 ){
			perror("The likelihood failed to increase!");
		}

		Iteration = iter;
		if (abs(loglik(iter - 1) - loglik(iter - 2)) < 1e-10) {

			break;
		}
	}

	Cm = sigma2y / sigma2beta + dd;
	Cs = 1 / sigma2beta + dd / sigma2y;
	Sigb = uu*diagmat(1 / Cs)*uu.t();
	mub = uu * (utxty / Cm);

	vec loglik_out;
	int to = Iteration -1;
	loglik_out = loglik.subvec(0, to);

 	loglik_max = loglik(to);
	iteration = Iteration -1;
	colvec var_comp(2);
	var_comp(0) = sigma2y;
	var_comp(1) = sigma2beta;
	return(var_comp);
}



double loss_gradient_lsq(colvec y,        // observed data // HL
			 arma::mat A,        // design matrix // HL
			 colvec x,        // effect sizes // HL
			 double lambda,           // regularization param. // HL
			 vec gradient, // gradient is stored here // HL
			 bool lasso = false) {    // indicator of L1/L2 penalty // HL
  int n = size(A)(0);
  int p = size(A)(1);
  double loss = 0, res = 0, Ax = 0;
  int i, j;
  for(j=0; j < p; ++j) gradient[j] = 0;    // initialize gradient
  for(i=0; i < n; ++i) {
    Ax = 0;
    for(int j=0; j < p; ++j)
      Ax += (A(i,j) * x[j]);
    res = y[i] - Ax;                       // compute residual per sample // HL
    loss += (res*res);                     // compute loss function // HL
    for(int j=0; j < p; ++j)
      gradient[j] += (-2 * A(i,j) * res);  // compute gradient // HL
  }                                        // end of loop over samples

  // Routine to reflect L1/L2 penalty
  if ( lambda > 0 ) {                      // skip if lambda is zero // HL
    if ( lasso ) {                         // L1 penalty is used if lasso == true // HL
      for(j=0; j < p; ++j) {
        gradient[j] += (lambda * (x[j] < 0 ? -1.0 : 1.0)); // update gradient // HL
        loss += (lambda * fabs(x[j]));                     // updaye loss // HL
      }
    }
    else {                                 // L2 penalty is ued if lasso == false // HL
      for(j=0; j < p; ++j) {
        gradient[j] += (2 * lambda * x[j]);  // update gradient // HL
        loss += (lambda * x[j] * x[j]);      // update loss // HL
      }
    }
  }

  for(j=0; j < p; ++j) gradient[j] /= n;   // objective function is scaled to 1/n
   
  return loss / n;                         // return (scaled) loss  
}
vec gradient_descent_lsq_initial(arma::vec y,         
				   arma::mat A,
				   arma::vec x0,
				   double lambda,
				   double gamma,
				   bool lasso = false,
				   int max_iter = 1000,
				   double tol = 1e-5) {
  int n = y.size();
  int p = x0.size();

  arma::vec gr(p);
  arma::vec xt = x0;
  double loss = loss_gradient_lsq(y, A, xt, lambda, gr, lasso); // evaluate initial gradient 
  double prevLoss = loss;                                       // store initial loss 
  int iter, j;

  // performs gradient descent with adaptive step size    // HL
  for(iter=1; iter < max_iter; ++iter) {
    for(j=0; j < p; ++j)
      xt[j] -= (gamma * gr[j]);                            // update params 

    loss = loss_gradient_lsq(y, A, xt, lambda, gr, lasso); // evaluate gradient and loss 

    if ( prevLoss > loss ) {                                   // if loss decreased // HL
      if ( ( prevLoss - loss ) / fabs(prevLoss) < tol ) break; // check convergence
      gamma *= 1.1;                                            // increase step size by 10% // HL
    }
    else gamma /= 2.0;                                     // if loss increased half the step size // HL
    prevLoss = loss;    
  }

  return(xt);  
}


//' @author Boran Gao, \email{borang@umich.edu}
//' @title
//' CoMM Binary
//' @description
//' CoMM Binary to dissecting genetic contributions to complex disease by leveraging regulatory information.
//'
//' @param Y  trait vector.
//' @param Z  gene expression vector.
//' @param X_1  normalized genotype (cis-SNPs) matrix for eQTL.
//' @param X_2  normalized genotype (cis-SNPs) matrix for GWAS.
//' @param constr  indicator for constraint (alpha = 0 if constr = 1).
//'
//' @return List of model parameters
//' @export
// [[Rcpp::export]]
List CoMM_Binary(arma::colvec &Y, arma::colvec &Z, arma::mat &X_1, arma::mat &X_2,double constr) {
	
	/*****************************************************************************
								Initialization Part
	*****************************************************************************/	

  int n1 = size(X_1)(0), p = size(X_1)(1), n2 = size(X_2)(0);

  double sigma2y, sigma2beta, loglik;
  mat Sigb = zeros<mat>(p,p);
  vec mub  = zeros<vec>(p),initial_var;
  int iter,maxIter=100;
  initial_var = lmm_pxem_ptr(Z,X_1, maxIter,sigma2y,sigma2beta,loglik,iter,Sigb,mub);
  //Rcout<<"                                                       "<<endl;
  Rcout<<"Initialization of Variance(VC ANALYSIS) Done."<<endl;
  //Rcout<<"                                                       "<<endl;
  double sigma_g_square = initial_var(1) * p, sigma_e_square =  initial_var(0);
  double sigma_square = var(Z);
  
  cout<<"Initial Ve Estimate is "<<sigma_e_square<<endl<<"Initial Vg Estimate is "<<sigma_g_square<<endl<<"Initial Sigma Squre is " << sigma_square <<endl;
 

  arma::colvec coef_initial = randn(p);
  wall_clock timer;
  timer.tic();
  arma::colvec coef= gradient_descent_lsq_initial(Z,X_1,coef_initial,5,0.01);
 // arma::colvec coef = randn(p) * sqrt(sigma_g_square/p);
  double beta_timer = timer.toc();
  
  //Rcout<<"                                                       "<<endl;
  Rcout<<"Initialization of Beta(LASSO) Done."<<endl;
 // Rcout<<"                                                       "<<endl;
  if(coef.has_nan() == TRUE){
	  coef = randn(p) * sqrt(sigma_g_square/p);
  }
  colvec x_2_beta = X_2 * coef;

  
  colvec alpha0= randn(1);
  double alpha = logisticGradientDescent(Y, x_2_beta, alpha0, 0.1, 1000 , 1e-8 ); // Get Initial Alpha Estimate
 // double alpha = 0;
  if(std::isfinite(alpha) == FALSE){
	  alpha = 0;
  }
    
 // Rcout<<"                                                       "<<endl;
  Rcout<<"Initialization of Alpha Done."<<endl;
 // Rcout<<"                                                       "<<endl;
  Rcout<< "Initial Alpha is "<<alpha <<endl;
  
  	/*****************************************************************************
								EM Part
	*****************************************************************************/	
  
  colvec err_1= randn(n2) * sqrt(sigma_square/n2),eta(n2),mu(n2),g_prime_mu(n2),y_tulta(n2),V_vec(n2),D_vec(n2),H_vec(n2),H_inv_vec(n2),mu_g(p),res_y(n2),res_z(n1);
 // err_1.zeros();
  mat t_X_2 = X_2.t();
  mat x_1_trans_x_1 = X_1.t() * X_1;
  mat Identity_mat = eye<mat>(n2,n2);
  mat Identity_mat_p= eye<mat>(p,p);
  mat inv_info(2,2);
  double prev_llk = -1e30, cur_llk,tol =1e-5, gamma = 1.0,step_size = 1,cur_llk_test_dev ,cur_llk_test; //gamma is the expansion parameter
  int max_iter = 200;
  iter = 0;
    
  for(iter = 0; iter < max_iter; iter++){
    
	eta = alpha*x_2_beta + err_1;
	mu = exp(eta)/(1+exp(eta));
	g_prime_mu = 1.0/(mu % (1-mu));
	y_tulta = eta + g_prime_mu % (Y - mu);// n*1
	D_vec = mu % (1-mu);
	V_vec.fill(sigma_square);
	H_vec = g_prime_mu + V_vec;
	H_inv_vec = 1/H_vec;
    
    mat x_2_H_inv = t_X_2.each_row() % H_inv_vec.t();
    mat x_2_H_inv_x_2 = x_2_H_inv * X_2;
    rowvec y_tulta_H_inv_X_2 = y_tulta.t()*x_2_H_inv.t();
    mat SIGMA_inv = gamma*gamma/sigma_e_square*x_1_trans_x_1+alpha*alpha*x_2_H_inv_x_2+p/sigma_g_square*Identity_mat_p;
	mat R = chol(SIGMA_inv);
	mat inv_R = inv(R);
	mat SIGMA_G = inv_R*inv_R.t();
    mat x_1_trans_x_1_SIGMA_G = x_1_trans_x_1 * SIGMA_G;
    mu_g =  SIGMA_G * trans(gamma/sigma_e_square*Z.t()*X_1+alpha*y_tulta_H_inv_X_2);
	res_y = y_tulta - alpha * X_2 * mu_g;
	res_z = Z-gamma*X_1*mu_g;
    if (constr == 1){
      alpha = 0;
    }
    else {
      alpha =  as_scalar(y_tulta_H_inv_X_2 * mu_g)/((as_scalar(mu_g.t() * x_2_H_inv_x_2* mu_g))+trace(x_2_H_inv_x_2*SIGMA_G));
    }
	
	
  //cur_llk = -0.5*log(det(H_mat*D_mat)) - 0.5 *  as_scalar(trans(y_tulta - alpha * X_2 * mu_g) * H_inv * (y_tulta - alpha * X_2 * mu_g)) - 0.5 * n1 * log(sigma_e_square) - 0.5/sigma_e_square *as_scalar(trans(Z-gamma*X_1*mu_g)*(Z-gamma*X_1*mu_g)) - 0.5* p *log(sigma_g_square) - 0.5/sigma_g_square *as_scalar(mu_g.t()*mu_g);
  //  Rcout<<"mu_g 0 is "<<mu_g(0)<<"mean is "<<mean(mu_g)<<" Varianc is "<<var(mu_g)<<std::endl;
  //   double sigma_square_gr =  -0.5 * trace(inv(Identity_mat+sigma_square * D_mat)*D_mat) + 0.5 * as_scalar(trans(y_tulta - alpha * X_2 * mu_g) * H_inv * H_inv * (y_tulta - alpha *X_2 * mu_g)) + alpha*alpha *trace(X_2.t()*H_inv * H_inv *X_2*SIGMA_G);// Use gradient descent method to update
  //  Rcout<<" sigma_square_gr " <<sigma_square_gr<<std::endl;
  //   sigma_square = sigma_square - step_size * sigma_square_gr/n2;   
 
 
 
  /*************************************************************************
							Expectation Step
  *************************************************************************/
    cur_llk = -0.5* sum(log(H_vec % D_vec)) - log(sigma_e_square)*n1 *0.5  - 0.5 *  sum(y_tulta % H_inv_vec % y_tulta)  - 0.5/sigma_e_square *sum(Z%Z) + sum (log(R.diag())) + 0.5*as_scalar(mu_g.t()*SIGMA_inv*mu_g);

    if(cur_llk>prev_llk){
      if((cur_llk - prev_llk)/abs(cur_llk) < tol){
        break;
      }else{
        step_size*= 1.5;
      }
    }else{
      step_size/= 2;
    }
    prev_llk = cur_llk;
	
  /*************************************************************************
						Maximization Step
  *************************************************************************/
	
    res_y = y_tulta - alpha * (X_2 * mu_g);
	vec H_inv_vec_2 = H_inv_vec % H_inv_vec;
	vec H_inv_vec_3 = H_inv_vec_2 % H_inv_vec;
	mat x_2_H_inv_2 = t_X_2.each_row() % H_inv_vec_2.t();
	mat x_2_H_inv_3 = t_X_2.each_row() % H_inv_vec_3.t();
    mat x_2_H_inv_2_x_2 = x_2_H_inv_2 * X_2;
	mat x_2_H_inv_2_x_3 = x_2_H_inv_3 * X_2;
	double tr_x_2_H_inv_2_x_2_SIGMA_G = trace(x_2_H_inv_2_x_2 * SIGMA_G);
	
	
	/***********************************************************************************************************************************
    colvec first_derivative(2);
    
    first_derivative(0) = as_scalar(mu_g.t() * x_2_H_inv * res_y) - alpha * trace(x_2_H_inv_x_2 * SIGMA_G);
    first_derivative(1) = -0.5 * sum(H_inv_vec) + 0.5 * sum(res_y%res_y%H_inv_vec_2) + 0.5 * alpha * alpha * tr_x_2_H_inv_2_x_2_SIGMA_G;
    //Second Order Derivatives
    
    mat second_derivative(2,2);
    // For alpha
    second_derivative(0,0) = -1.0 * as_scalar(mu_g.t() * x_2_H_inv_x_2 * mu_g) - trace(x_2_H_inv_x_2*SIGMA_G);
    second_derivative(0,1) =  -1.0 * as_scalar(mu_g.t() * x_2_H_inv_2 * res_y) + alpha * tr_x_2_H_inv_2_x_2_SIGMA_G;
    
    //For Sigma square
    
    second_derivative(1,0) = -1.0 * as_scalar(mu_g.t() * x_2_H_inv_2 *res_y) + alpha * tr_x_2_H_inv_2_x_2_SIGMA_G;
    second_derivative(1,1) = 0.5*sum(H_inv_vec_2) - sum(res_y % res_y % H_inv_vec_3) - pow(alpha,2)*trace(x_2_H_inv_2_x_3*SIGMA_G);
    
    // Update
    colvec update = inv(second_derivative) * first_derivative;
    inv_info = inv(second_derivative);
   
    sigma_square-=update(1)*0.1;
	****************************************************************************************************************************/
	
	
	
	/************************************Test Case***********************************************************************************************/
	mat H_update = alpha*alpha* X_2 * SIGMA_G * X_2.t() + res_y * res_y.t();
	
	V_vec = H_update.diag() - g_prime_mu;
	sigma_square = mean(V_vec);
	
    x_2_beta = X_2 * mu_g;
    err_1 = sigma_square * res_y % H_inv_vec;  
	double tr_x_1_trans_x_1_SIGMA_G = trace(x_1_trans_x_1_SIGMA_G);
    sigma_e_square =  (sum(res_z%res_z) + gamma*gamma*tr_x_1_trans_x_1_SIGMA_G)/n1 ;
    gamma = 1.0/(as_scalar(mu_g.t() * x_1_trans_x_1 * mu_g) + tr_x_1_trans_x_1_SIGMA_G ) * as_scalar(Z.t()*X_1*mu_g);
    sigma_g_square = (as_scalar(mu_g.t()*mu_g) + trace(SIGMA_G));
	

  //First Order Derivativs
  //  cur_llk_alpha_0 = -0.5*log(det(H_mat*D_mat)) - 0.5 *  as_scalar(trans(y_tulta) * H_inv * (y_tulta )) - 0.5 * n1 * log(sigma_e_square) - 0.5/sigma_e_square *as_scalar(trans(Z-gamma*X_1*mu_g)*(Z-gamma*X_1*mu_g)) - 0.5* p *log(sigma_g_square) - 0.5/sigma_g_square *as_scalar(mu_g.t()*mu_g)-n2;
  // cur_llk_test= -0.5*log(det(H_mat*D_mat)) - 0.5 *  as_scalar(trans(y_tulta - alpha * X_2 * mu_g) * H_inv * (y_tulta - alpha * X_2 * mu_g)) - 0.5 * n1 * log(sigma_e_square) - 0.5/sigma_e_square *as_scalar(trans(Z-gamma*X_1*mu_g)*(Z-gamma*X_1*mu_g)) - 0.5* p *log(det(SIGMA_G)) - 0.5*as_scalar(mu_g.t()*SIGMA_G*mu_g);
  // cur_llk_test = -0.5*log(det(H_mat*D_mat)) - 0.5 *  as_scalar(trans(y_tulta) * H_inv * (y_tulta ))  - 0.5/sigma_e_square *as_scalar(trans(Z)*(Z)) - sum (log(R.diag())) + 0.5*as_scalar(mu_g.t()*SIGMA_G*mu_g);
	res_z = Z - gamma*X_1*mu_g;
	double Eab = 0.5 *  sum(res_y%H_inv_vec%res_y) + 0.5/sigma_e_square *sum(res_z%res_z) +  0.5/sigma_g_square*sum(mu_g%mu_g);
	cur_llk_test = -log(sigma_g_square)*p*0.5 -0.5* sum(log(H_vec % D_vec)) - log(sigma_e_square)*n1 *0.5 - sum(log(R.diag())) - Eab;
	cur_llk_test_dev = -0.5* sum(log(H_vec % D_vec)) - log(sigma_e_square)*n1 *0.5  - 0.5 *  sum(y_tulta % H_inv_vec % y_tulta)  - 0.5/sigma_e_square *sum(Z%Z) + sum (log(R.diag())) + 0.5*as_scalar(mu_g.t()*SIGMA_inv*mu_g);

  //    cout << "mean = " << endl << mean(err_1) << endl;
  //    cout << "var  = " << endl << var(err_1)  << endl;
 //     cout << "range  = " << endl << range(err_1)  << endl;
//	double Eab = 0.5 *  sum(y_tulta%H_inv_vec%y_tulta) + 0.5/sigma_e_square *sum(Z%Z) - 0.5*as_scalar(mu_g.t()*SIGMA_inv*mu_g);
 //   cur_llk_test = -0.5* sum(log(H_vec % D_vec)) - log(sigma_e_square)*n1 *0.5 - sum(log(R.diag())) -Eab;
  //  Rcout<<"alpha is "<<alpha<<" Gamma is "<<gamma<<" sigma_e_square is "<< sigma_e_square  << "sigma square is "<<sigma_square<<" Sigma square update is  "<<update(1)<<std::endl;
	// Rcout<<"alpha is "<<alpha<<" sigma_e_square is "<< sigma_e_square <<" sigma_g_square is "<< sigma_g_square<< "sigma square is "<< sigma_square <<std::endl;
//	Rcout<< "Log Likelihood is "<<cur_llk_test<<std::endl;
  }
  
  // double lrt = cur_llk/cur_llk_alpha_0;
  return(List::create(Named("sigma_e_square") = sigma_e_square,Named("alpha")=alpha,Named("sigma_square") = sigma_square,Named("sigma_g_square") = sigma_g_square,Named("alpha_sd") = sqrt(-inv_info(0,0)),Named("LRT")=cur_llk_test,Named("LRT_Dev")=cur_llk_test_dev,Named("Num_Iter")=iter));
}






