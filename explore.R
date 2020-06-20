

parameters$W4
learning_rate<-0.02
# X<-train_x
# Y<-train_y
# layers_dims<-c(12288, 20, 7, 5, 1)
# learning_rate<-0.0075

#parameters<-initialize_parameters_deep(layers_dims)
##################################
for(i in 1:5000){
  # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
  #AL, caches = L_model_forward(X, parameters)
  AL_caches<-L_model_forward(X, parameters)
  AL<-AL_caches[["AL"]]
  caches<-AL_caches[["caches"]]

  #
  #
  caches<-list()
  A<-X
  L<-as.integer(length(parameters)/2)
  
  for (l in 1:(L-1)){
    A_prev<-A
    A_cache<-linear_activation_forward(A_prev, parameters[[sprintf("W%s",l)]], parameters[[sprintf("b%s",l)]], activation="relu")
    A<-A_cache$A
    caches[[l]]<-A_cache$cache
  }
  ZL<-linear_forward(A, parameters[[sprintf("W%s",L)]], parameters[[sprintf("b%s",L)]])[["Z"]]
  AL_cache = linear_activation_forward(A, parameters[[sprintf("W%s",L)]], parameters[[sprintf("b%s",L)]], activation="sigmoid")
  AL<-AL_cache$A
  caches[[L]]<-AL_cache$cache
  stopifnot(dim(AL)==c(1,dim(X)[2]))
  return(list(AL=AL,caches=caches))
  
  # AL[1,1:5] # OK
  # 
  # #show A in cache
  # caches[[1]][["linear_cache"]][["A"]]
  # # show Z in cache
  # caches[[1]][["activation_cache"]]
  # # show W (returns W1 from  
  # caches[[1]][["linear_cache"]][["W"]][1,1:10]
  # # show b in cache
  # caches[[1]][["linear_cache"]][["b"]]
  
  # Compute cost.
  cost<-compute_cost(AL, Y)
  print(cost) # OK
  summary(as.vector(AL))
  log(AL)%*%t(Y)
  log(1-AL)%*%(1-t(Y))
  
  # Backward propagation.
  grads<-L_model_backward(AL, Y, caches)
  
  # #dW1
  # grads[[1]][["dW1"]][1:5,1:5] # OK
  # #db1
  # grads[[1]][["db1"]] # OK
  # #dW4
  # grads[[4]][["dW4"]][1,1:5] #OK
  # #db4
  # grads[[4]][["db4"]] # OK
  
  # Update parameters.
  parameters<-update_parameters(parameters, grads, learning_rate)
  # parameters$W4 # OK
  # parameters$W3 # OK
  # parameters$W2[1:5,1:5] # OK
  # parameters$W1[1:5,1:5] # OK
  # parameters$b4 # OK
  # parameters$b3 # OK
  # parameters$b2 # OK
}

AL_caches<-L_model_forward(X, parameters)
AL<-AL_caches[["AL"]]
table(train_y==(AL>.5))

AL_caches_t<-L_model_forward(test_x, parameters)
AL_t<-AL_caches_t[["AL"]]
table(test_y==(AL_t>.5))

######################

m=dim(Y)[2]
cost<-unname((-1/m)*(log(AL)%*%t(Y)+log(1-AL)%*%(1-t(Y)))[1,1])
summary(as.vector(AL))
log(AL)%*%t(Y)
tt<-log(1-AL)*(1-Y)
which(tt,is.nan(tt))
AL[,8137]
Y[,8137]

AL[,which(AL==1)]
which.max(AL)
