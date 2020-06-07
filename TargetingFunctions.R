featureNormalize2<-function(xm, infl){
  
  scalingMatrix<-apply(xm,2,function(x){
    cbind(mean(x),sd(x))
  })
  rownames(scalingMatrix)<-c("mu","sigma")
  
  # if model has constant, do not scale the constant
  if(infl==1){
    scalingMatrix[1,1]<-0
    scalingMatrix[2,1]<-1
  }
  
  # if 0 all across
  scalingMatrix[1,scalingMatrix[1,]==0]<-0
  scalingMatrix[2,scalingMatrix[1,]==0]<-1
  # #
  
  drop_quotes<-sub("`","",colnames(scalingMatrix))
  drop_quotes<-sub("`","",drop_quotes)
  scalingMatrix[1,grep("^CD|^SZ|^KG|^EA|^Probabi",drop_quotes)]<-0
  scalingMatrix[2,grep("^CD|^SZ|^KG|^EA|^Probabi",drop_quotes)]<-1
  #
  
  
  mus<-scalingMatrix[rep(1,nrow(xm)),]
  sds<-scalingMatrix[rep(2,nrow(xm)),]
  X_norm<-(xm-mus)/sds
  out<-list(x=X_norm,scalingMatrix=scalingMatrix)
}


sigmoid<-function(z){
  g<-1/(1+exp(-z))
}


ComputeCostGradient<-function(type){
  
  function(X, y, theta, infl, lambda){
    # length of theta or design matrix
    n<-dim(X)[2]
    # number of observations
    m<-dim(X)[1]
    # vector to drop Xo from regularization component. check if Xo supplied to the model formula
    if(infl==1) 
      lambda_vector<-c(0,rep(1,n-1))
    else 
      lambda_vector<-c(rep(1,n))
    
    if(type=="J"){
      (-1/m)*sum(y*log(sigmoid(X%*%theta))+(1-y)*log(1-sigmoid(X%*%theta)))+lambda/(2*m)*lambda_vector%*%theta^2
    }
    else if(type=="grad"){
      #as.numeric((1/m)*t(X)%*%(sigmoid(X%*%theta)-y)+lambda/m*lambda_vector*theta)
      (1/m)*(t(X)%*%(sigmoid(X%*%theta)-y))+lambda/m*lambda_vector*theta
    }
    else stop("Invalid output request from CostGradient: acceptable values are: 'J' and 'grad'")
  }
}


gdlreg2<-function(formula,data,subset,theta, lambda=0, method ="Nelder-Mead", normalize=T){
  if(is.na(match(method,c("Nelder-Mead", "BFGS", "L-BFGS-B", "CG")))) 
    stop("Please select on of the tested methods: Nelder-Mead, BFGS, L-BFGS-B", "CG")
  
  mf <- match.call(expand.dots = F)
  m <- match(c("formula", "data","subset"), 
             names(mf), 0L)
  mf <- mf[c(1L, m)]
  #mf <- mf[m]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  
  y <- model.response(mf, "numeric")
  x <- model.matrix(mt, mf)
  # flag if intercept was selected. will set lambda vector[1] (regularization variable) to zero if intercept present in formula
  infl <- attr(mt,"intercept")
  #
  n<-ncol(x)
  if(n!=length(theta)) 
    stop("Model formula and initial theta have incompatible dimensions")
  scalingMatrix<-NULL
  if(normalize){
    normal_list<-featureNormalize2(x,infl=infl)
    x<-normal_list$x
    scalingMatrix<-normal_list$scalingMatrix
  }
  
  # two closures are created by ComputeCostGradient function for J and grad
  J<-ComputeCostGradient("J")
  grad<-ComputeCostGradient("grad")
  
  # optimize based on advanced algorithm. same as Octave's fminunc
  res_all<-optim(theta,J,grad,X=x,y=y,infl=infl,lambda=lambda,method=method)
  res<-res_all$par
  conv<-res_all$convergence
  names(res)<-colnames(x)
  res
  model<-list(theta=res, scalingMatrix=scalingMatrix, convergence=conv)
}


##################### predict LR #######################

predictLR<-function(theta_scaling, X){
  theta<-theta_scaling[["theta"]]
  scalingMatrix<-theta_scaling[["scalingMatrix"]]
  if("(Intercept)" %in% names(theta)){
    X<-data.table(temp=1,X)
    names(X)[1] <- "(Intercept)"
  }
  
  if(is.null(scalingMatrix)){
    X_norm<-X
  } else {
    
    # scale
    mus<-scalingMatrix[rep(1,nrow(X)),]
    sds<-scalingMatrix[rep(2,nrow(X)),]
    X_norm<-(X-mus)/sds
  }
  #scale
  
  
  
  sigmoid(as.matrix(X_norm)%*%theta)
}

# save object into it's modelBrand Folder
savePredict<-function(obj,textadd="Target List"){
  
  predict_dir_path<-file.path(getwd(),modelBrand)
  predict_file_name<-file.path(predict_dir_path,paste0(Sys.Date()," ",modelBrand," ",predictSeason," ",textadd,".xlsx"))
  
  if(dir.exists(predict_dir_path))
    write.xlsx(obj,predict_file_name)
  else{
    dir.create(predict_dir_path)
    write.xlsx(obj,predict_file_name)
  }
  
}
