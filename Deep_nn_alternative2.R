source("CommonNN.R")
# 3.2 initialize parameters deep

initialize_parameters_deep<-function(layer_dims){
  set.seed(12) #comment out
  parameters<-list()
  L<-length(layer_dims)
  for(l in 2:L){
    #parameters[[sprintf("W%s",l-1)]]<-matrix(rnorm(layer_dims[l-1]*layer_dims[l])*.01,ncol=layer_dims[l-1])
    parameters[[sprintf("W%s",l-1)]]<-matrix(rnorm(layer_dims[l-1]*layer_dims[l]),ncol=layer_dims[l-1])/sqrt(layer_dims[l-1])
    parameters[[sprintf("b%s",l-1)]]<-ones_zeros(0,c(layer_dims[l],1))
  }
  return(parameters)
}


# test
parameters<-initialize_parameters_deep(c(5,4,3))

# 4.1 Linear forward

linear_forward<-function(A, W, b){
  
  # Arguments:
  # A -- activations from previous layer (or input data): (size of previous layer, number of examples)
  # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  # b -- bias vector, numpy array of shape (size of the current layer, 1)
  # 
  # Returns:
  # Z -- the input of the activation function, also called pre-activation parameter 
  # cache -- a list  containing "A", "W" and "b" ; stored for computing the backward pass efficiently
  # 
  b_broadcast<-dim(A)[2]
  #b_calc used to broadcast
  b_calc<-b[,rep(1,times=b_broadcast)]
  Z<-W%*%A+b_calc
  stopifnot(dim(Z)==c(dim(W)[1],dim(A)[2]))
  return(list(Z=Z,cache=list(A=A,W=W,b=b)))
}

# test
A<-matrix(c(1.62434536, -0.61175641,
            -0.52817175, -1.07296862,
            0.86540763, -2.3015387 ),nrow=3,byrow = T)
W<-matrix(c(1.74481176, -0.7612069 ,  0.3190391),nrow=1,byrow=T)
b<-matrix(-0.24937038,nrow=dim(W)[1])
#b<-b[,rep(1,times=dim(A)[2])]

tt<-linear_forward(A, W, b)
tt

# 4.2 linear activation forward
linear_activation_forward<-function(A_prev, W, b, activation){
  if(activation=="sigmoid"){
    
    Z_linear_cache<-linear_forward(A_prev, W, b)
    Z<-Z_linear_cache[["Z"]]
    linear_cache<-Z_linear_cache[["cache"]]
    
    A_activation_cache<-sigmoid(Z)
    A<-A_activation_cache[["A"]]
    activation_cache<-A_activation_cache[["cache"]]
  } 
  if (activation=="relu"){
    Z_linear_cache<-linear_forward(A_prev, W, b)
    Z<-Z_linear_cache[["Z"]]
    linear_cache<-Z_linear_cache[["cache"]]
    
    A_activation_cache<-relu(Z)
    A<-A_activation_cache[["A"]]
    activation_cache<-A_activation_cache[["cache"]]
  }
  stopifnot(dim(A)==c(dim(W)[1],dim(A_prev)[2]))
  cache<-list(linear_cache=linear_cache,activation_cache=activation_cache)
  return(list(A=A,cache=cache))
}

# test
A_prev<-matrix(c(-0.41675785, -0.05626683,-2.1361961 ,  1.64027081,-1.79343559, -0.84174737),ncol=2,byrow=T)
W<-matrix(c(0.50288142, -1.24528809, -1.05795222),ncol=3,byrow=T)
b<-matrix(-0.90900761,nrow=dim(W)[1])
#b<-b[,rep(1,times=dim(A_prev)[2])]
# adding to list
tt<-linear_activation_forward(A_prev, W, b, "relu")
tt


# linear_activation_forward(A_prev, W, b, "sigmoid")
# tt<-linear_activation_forward(A_prev, W, b, "relu")

L_model_forward<-function(X, parameters){
  caches<-list()
  A<-X
  L<-as.integer(length(parameters)/2)
  
  for (l in 1:(L-1)){
    A_prev<-A
    A_cache<-linear_activation_forward(A_prev, parameters[[sprintf("W%s",l)]], parameters[[sprintf("b%s",l)]], activatio="relu")
    A<-A_cache$A
    caches[[l]]<-A_cache$cache
  }
  AL_cache = linear_activation_forward(A, parameters[[sprintf("W%s",L)]], parameters[[sprintf("b%s",L)]], activation="sigmoid")
  AL<-AL_cache$A
  caches[[L]]<-AL_cache$cache
  stopifnot(dim(AL)==c(1,dim(X)[2]))
  return(list(AL=AL,caches=caches))
}

#test
X<-matrix(c(-0.31178367,  0.72900392,  0.21782079, -0.8990918,
            -2.48678065,  0.91325152,  1.12706373, -1.51409323,
            1.63929108, -0.4298936 ,  2.63128056,  0.60182225,
            -0.33588161,  1.23773784,  0.11112817,  0.12915125,
            0.07612761, -0.15512816,  0.63422534,  0.810655),nrow=5,byrow=T)

W1<-matrix(c( 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384,
              -1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953,
              -0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143,
              -0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059),nrow=4,byrow=T)


b1<-matrix(c(1.38503523,
             -0.51962709,
             -0.78015214,
             0.95560959),ncol=1,byrow=T)

W2<-matrix(c(-0.12673638, -1.36861282,  1.21848065, -0.85750144,
             -0.56147088, -1.0335199 ,  0.35877096,  1.07368134,
             -0.37550472,  0.39636757, -0.47144628,  2.33660781),nrow=3,byrow=T)

b2<-matrix(c( 1.50278553,
              -0.59545972,
              0.52834106),ncol=1,byrow=T)

W3<-matrix(c( 0.9398248 ,  0.42628539, -0.75815703),nrow=1,byrow=T)
b3<-matrix(c(-0.16236698),ncol=1,byrow=T)

parameters<-list(W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)


tt<-L_model_forward(X, parameters)
tt
#cache_temp<-L_model_forward(X, parameters)[["caches"]][1][[1]][["linear_cache"]]
# 5.0 compute cost

compute_cost<-function(AL, Y){
  m=dim(Y)[2]
  cost<-unname((-1/m)*(log(AL)%*%t(Y)+log(1-AL)%*%(1-t(Y)))[1,1])
  stopifnot(is.null(dim(cost)))
  return(cost)
}

#test
Y<-matrix(c(1, 1, 0),nrow=1,byrow=T)
AL<-matrix(c(0.8, 0.9, 0.4),nrow=1,byrow=T)
(compute_cost(AL,Y))

################## backward #####################

# 6.1 linear backward

linear_backward<-function(dZ,cache){
  A_prev<-cache$A
  W<-cache$W
  b<-cache$b
  m<-dim(A_prev)[2]
  
  ######### 3 equations
  dW<-dZ%*%t(A_prev)/m
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

# test
linear_cache<-list(
  A=matrix(c(-0.3224172 , -0.38405435,  1.13376944, -1.09989127,
             -0.17242821, -0.87785842,  0.04221375,  0.58281521,
             -1.10061918,  1.14472371,  0.90159072,  0.50249434,
             0.90085595, -0.68372786, -0.12289023, -0.93576943,
             -0.26788808,  0.53035547, -0.69166075, -0.39675353), nrow=5,byrow=T),
  
  W=matrix(c(-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035,
             0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896,
             -0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548),nrow=3,byrow=T),
  
  b=matrix(c(2.10025514,
             0.12015895,
             0.61720311),ncol=1,byrow=T)
)

dZ<-matrix(c(1.62434536, -0.61175641, -0.52817175, -1.07296862,
             0.86540763, -2.3015387 ,  1.74481176, -0.7612069,
             0.3190391 , -0.24937038,  1.46210794, -2.06014071),nrow=3,byrow=T)

linear_backward(dZ,linear_cache)

# 6.2 linear activation backward

linear_activation_backward<-function(dA,cache,activation){
  linear_cache<-cache$linear_cache
  activation_cache<-cache$activation_cache
  if(activation=="relu"){
    dZ<-relu_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward(dZ, linear_cache)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  if(activation=="sigmoid"){
    dZ<-sigmoid_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward(dZ, linear_cache)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  return(list(dA_prev=dA_prev, dW=dW, db=db))
}

# test
cache<-list(linear_cache=list(A=matrix(c(-2.1361961 ,  1.64027081,
                                         -1.79343559, -0.84174737,
                                         0.50288142, -1.24528809),nrow=3, byrow=T),
                              
                              W=matrix(c(-1.05795222, -0.90900761,  0.55145404),nrow=1, byrow=T),
                              
                              b=matrix(c(2.29220801),nrow=1,byrow=T)),
            
            activation_cache=matrix(c(0.04153939, -1.11792545),nrow=1,byrow=T))

dAL<-matrix(c(-0.41675785, -0.05626683),nrow=1,byrow=T)
tt2<-linear_activation_backward(dAL,cache,"relu")
names(tt2)<-c(sprintf("dA%s",L),sprintf("dW%s",L),sprintf("db%s",L))
tt2
# 6.3 L-Model backward
#tt[[1]]

L_model_backward<-function(AL,Y,caches){
  grads<-list()
  L<-length(caches)
  m<-dim(AL)[2]
  dAL<-(-1)*(Y/AL-((1-Y)/(1-AL)))
  current_cache<-caches[[L]]
  
  grads[[L]]<-linear_activation_backward(dAL,current_cache,"sigmoid")
  names(grads[[L]])<-c(sprintf("dA%s",L-1),sprintf("dW%s",L),sprintf("db%s",L))
  
  for(l in rev(1:(L-1))){
    current_cache<-caches[[l]]
    grads[[l]]<-linear_activation_backward(grads[[l+1]][[sprintf("dA%s",l)]],current_cache,"relu")
    names(grads[[l]])<-c(sprintf("dA%s",l-1),sprintf("dW%s",l),sprintf("db%s",l))
  }
  return(grads)
}

# test

AL<-matrix(c(1.78862847, 0.43650985),nrow=1)
Y_assess<-matrix(c(1,0),nrow=1)
caches<-list(
  list(
    linear_cache=list(A=matrix(c(  0.09649747, -1.8634927  ,
                                   -0.2773882 , -0.35475898 ,
                                   -0.08274148, -0.62700068 ,
                                   -0.04381817, -0.47721803  ), nrow=4,byrow=T),
                      W=matrix(c( -1.31386475,  0.88462238,  0.88131804,  1.70957306 ,
                                  0.05003364, -0.40467741, -0.54535995, -1.54647732 ,
                                  0.98236743, -1.10106763, -1.18504653, -0.2056499   ), nrow=3,byrow=T),
                      b=matrix(c(  1.48614836 ,
                                   0.23671627 ,
                                   -1.02378514   ), nrow=3,byrow=T)),
    activation_cache=matrix(c( -0.7129932 ,  0.62524497 ,
                               -0.16051336, -0.76883635 ,
                               -0.23003072,  0.74505627   ) , nrow=3,byrow=T)),
  list(
    linear_cache=list(A=matrix(c(  1.97611078, -1.24412333 ,
                                   -0.62641691, -0.80376609 ,
                                   -2.41908317, -0.92379202  ), nrow=3,byrow=T),
                      W=matrix(c( -1.02387576,  1.12397796, -0.13191423  ), nrow=1,byrow=T),
                      b=matrix(c( -1.62328545   ), nrow=1,byrow=T)),
    activation_cache=matrix(c(  0.64667545, -0.35627076   ) , nrow=1,byrow=T))
)

grads<-L_model_backward(AL, Y_assess, caches)

# 6.4 Update parameters

update_parameters<-function(parameters, grads, learning_rate){
  L<-as.integer(length(parameters)/2)
  for(l in 1:L){
    parameters[[sprintf("W%s",l)]] = parameters[[sprintf("W%s",l)]] - learning_rate*grads[[l]][[sprintf("dW%s",l)]]
    parameters[[sprintf("b%s",l)]] = parameters[[sprintf("b%s",l)]] - learning_rate*grads[[l]][[sprintf("db%s",l)]]
  }
  return(parameters)
}

# test
#for(l in 1:L) print(l)
parameters<-list(W1=matrix(c(-0.41675785, -0.05626683, -2.1361961 ,  1.64027081,
                             -1.79343559, -0.84174737,  0.50288142, -1.24528809,
                             -1.05795222, -0.90900761,  0.55145404,  2.29220801), nrow=3,byrow=T),
                 b1=matrix(c( 0.04153939,
                              -1.11792545,
                              0.53905832), ncol=1,byrow=T),
                 W2=matrix(c(-0.5961597 , -0.0191305 ,  1.17500122), nrow=1,byrow=T),
                 b2=matrix(c(-0.74787095), nrow=1,byrow=T))

grads<-list(list(dW1=matrix(c( 1.78862847,  0.43650985,  0.09649747, -1.8634927  ,
                               -0.2773882 , -0.35475898, -0.08274148, -0.62700068 ,
                               -0.04381817, -0.47721803, -1.31386475,  0.88462238), nrow=3, byrow=T),
                 db1=matrix(c(0.88131804 ,
                              1.70957306 ,
                              0.05003364), ncol=1, byrow=T)),
            list(dW2=matrix(c(-0.40467741, -0.54535995, -1.54647732), nrow=1, byrow=T),
                 db2=matrix(c(0.98236743), nrow=1, byrow=T))
)

(update_parameters(parameters, grads, learning_rate=0.1))
