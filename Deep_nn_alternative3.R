# L2 regularization:
# cost
# backpropagation

# cost


compute_cost_with_regularization<-function(AL, Y, parameters, lambd){
  m<-dim(Y)[2]
  cost<-unname((-1/m)*(log(AL)%*%t(Y)+log(1-AL)%*%(1-t(Y)))[1,1])
  stopifnot(is.null(dim(cost)))
  
  L<-as.integer(length(parameters)/2)
  L2_component<-0
  for(l in 1:L){
    L2_temp<-sum(parameters[[sprintf("W%s",1)]]^2)
    L2_component<-L2_component+L2_temp
  }
  L2_component<-L2_component*(lambd/(2*m))
  cost<-cost+L2_component
  
  return(cost)
}

# 6.1 linear backward

linear_backward_with_regularization<-function(dZ,cache,lambd){
  A_prev<-cache$A
  W<-cache$W
  b<-cache$b
  m<-dim(A_prev)[2]
  
  ######### 3 equations
  dW<-dZ%*%t(A_prev)/m+lambd/m*W
  # broadcast only when using b in a formula
  #db_broadcast<-dim(b)[2]
  db<-matrix(rowSums(dZ),ncol=1,byrow=T)/m
  #db<-db[,rep(1,times=db_broadcast)]
  dA_prev<-t(W)%*%dZ
  ######### 3 equations
  
  stopifnot(dim(A_prev)==dim(dA_prev))
  stopifnot(dim(W)==dim(dW))
  stopifnot(dim(b)[1]==dim(db)[1])
  
  return(list(dA_prev=dA_prev,dW=dW,db=db))
}

# 6.2 linear activation backward

linear_activation_backward_with_regularization<-function(dA,cache,activation,lambd){
  linear_cache<-cache$linear_cache
  activation_cache<-cache$activation_cache
  if(activation=="relu"){
    dZ<-relu_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward_with_regularization(dZ, linear_cache, lambd)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  if(activation=="sigmoid"){
    dZ<-sigmoid_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward_with_regularization(dZ, linear_cache, lambd)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  return(list(dA_prev=dA_prev, dW=dW, db=db))
}

# 6.3 L-Model backward


L_model_backward_with_regularization<-function(AL,Y,caches,lambd){
  grads<-list()
  L<-length(caches)
  m<-dim(AL)[2]
  dAL<-(-1)*(Y/AL-((1-Y)/(1-AL)))
  current_cache<-caches[[L]]
  
  grads[[L]]<-linear_activation_backward_with_regularization(dAL,current_cache,"sigmoid",lambd)
  names(grads[[L]])<-c(sprintf("dA%s",L-1),sprintf("dW%s",L),sprintf("db%s",L))
  
  for(l in rev(1:(L-1))){
    current_cache<-caches[[l]]
    grads[[l]]<-linear_activation_backward_with_regularization(grads[[l+1]][[sprintf("dA%s",l)]],current_cache,"relu",lambd)
    names(grads[[l]])<-c(sprintf("dA%s",l-1),sprintf("dW%s",l),sprintf("db%s",l))
  }
  return(grads)
}

L_layer_model_with_regularization<-function(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=F,lambd=0){
  set.seed(1)
  cost<-NULL
  parameters<-initialize_parameters_deep(layers_dims)
  for(i in 1:num_iterations){
    
    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    #AL, caches = L_model_forward(X, parameters)
    AL_caches<-L_model_forward(X, parameters)
    AL<-AL_caches[["AL"]]
    caches<-AL_caches[["caches"]]
    
    # Compute cost.
    cost<-compute_cost_with_regularization(AL, Y, parameters, lambd)
    
    # Backward propagation.
    grads<-L_model_backward_with_regularization(AL, Y, caches,lambd)
    
    # Update parameters.
    parameters<-update_parameters(parameters, grads, learning_rate)
    #print(i)
    # print cost
    if(print_cost & (i %% 100) ==0){
      print(cost)
    }
  }
  return(parameters)
}