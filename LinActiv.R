#https://medium.com/analytics-vidhya/a-beginners-guide-to-learning-r-with-the-titanic-dataset-a630bc5495a8
# desision tree

library(openxlsx)
library(ggplot2)
library(magrittr)

# X<-read.xlsx("PlanarX.xlsx")
# Y<-read.xlsx("PlanarY.xlsx")
# names(X)<-paste0("V",names(X))
# names(Y)<-"color_cd"
# 
# ggplot(data=X)+geom_point(aes(x=V0,y=V1,color=as.factor(Y$color_cd)))+
#   scale_color_manual(values = c("red","blue"))+
#   labs(color="Type")+
#   theme(panel.background = element_rect(fill="grey50"))

set.seed(1)
# transpose matrix so that each observation is one column
# X<-t(as.matrix(X))
# Y<-t(as.matrix(Y))
# 
# shape_X<-dim(X)
# shape_Y<-dim(Y)

# layer sizes

layer_sizes<-function(X,Y){
  n_x<-dim(X)[1]
  n_h<-4
  n_y<-dim(Y)[1]
  return(c(n_x=n_x,n_h=n_h,n_y=n_y)) # pass to n_all
}

# test
# layer_sizes(X,Y)

# initialize parameters

initialize_parameters<-function(n_all){
  n_x<-n_all[1]
  n_h<-n_all[2]
  n_y<-n_all[3]
  W1<-matrix(rnorm(n_h*n_x)*.01,ncol=n_x)
  b1<-ones_zeros(0,c(n_h,1))
  W2<-matrix(rnorm(n_y,n_h)*.01,ncol=n_h)
  b2<-ones_zeros(0,c(n_y,1))
  return(list(W1=W1,b1=b1,W2=W2,b2=b2))
}

# test
# (initialize_parameters(layer_sizes(X,Y)))

lin_forward_propagation<-function(X,parameters){
  W1<-parameters$W1
  b1<-parameters$b1[,rep(1,times=dim(X)[2])]
  W2<-parameters$W2
  b2<-parameters$b2[,rep(1,times=dim(X)[2])]
  
  # Implement Forward Propagation to calculate A2 (probabilities)
  Z1<-W1%*%X+b1
  A1<-tanh(Z1)
  Z2<-W2%*%A1+b2
  #A2<-sigmoid(Z2)
  A2<-Z2
  #print(dim(A2)==c(1,dim(X)[2]))
  # cache
  return(list(Z1=Z1,A1=A1,Z2=Z2,A2=A2))
}

# Test


lin_compute_cost<-function(A2,Y){
  m<-dim(Y)[2]
  #cost<-unname((-1/m)*(log(A2)%*%t(Y)+log(1-A2)%*%(1-t(Y)))[1,1])
  cost<-unname((1/(2*m))*sum((A2-Y)^2))
}



# backward propagation

lin_backward_propagation<-function(parameters,cache,X,Y){
  m<-dim(Y)[2]
  # First, retrieve W1 and W2 from the dictionary "parameters".
  W1<-parameters$W1
  W2<-parameters$W2
  # Retrieve also A1 and A2 from dictionary "cache".
  A1<-cache$A1
  A2<-cache$A2
  Z2<-cache$Z2
  # Backward propagation: calculate dW1, db1, dW2, db2. 
  dZ2<-(1/m)*(Z2-Y)
  dW2<-dZ2%*%t(A1)/m
  db2<-matrix(rowMeans(dZ2))
  dZ1<-(t(W2)%*%dZ2)*(1-A1^2)
  dW1<-dZ1%*%t(X)/m
  db1<-matrix(rowMeans(dZ1))
  #grads
  return(list(dW1=dW1,dW2=dW2,db1=db1,db2=db2))
}



# update parameters
update_parameters<-function(parameters, grads, learning_rate){
  # Retrieve each parameter from the dictionary "parameters"
  W1 <- parameters[["W1"]]
  b1 <- parameters[["b1"]]
  W2 <- parameters[["W2"]]
  b2 <- parameters[["b2"]]
  ### END CODE HERE ###
  
  # Retrieve each gradient from the dictionary "grads"
  
  dW1 <- grads[["dW1"]]
  db1 <- grads[["db1"]]
  dW2 <- grads[["dW2"]]
  db2 <- grads[["db2"]]
  
  # Update rule for each parameter
  
  W1 <- W1-learning_rate*dW1
  b1 <- b1-learning_rate*db1
  W2 <- W2-learning_rate*dW2
  b2 <- b2-learning_rate*db2
  #parameters
  return(list(W1=W1,b1=b1,W2=W2,b2=b2))
}


#update_parameters(parameters, backward_propagation(parameters,cache,X,Y), learning_rate = 1.2)

lin_nn_model<-function(X, Y, n_h, num_iterations = 10000, print_cost=F,learning_rate = 1.2){
  set.seed(3)
  n_x<-layer_sizes(X, Y)[1]
  n_y<-layer_sizes(X, Y)[3]
  
  #initialize parameters
  parameters<-initialize_parameters(c(n_x,n_h,n_y))
  # gradient descent
  for (i in 1:num_iterations){
    # forward
    cache<-lin_forward_propagation(X,parameters)
    # cost
    cost<-lin_compute_cost(cache[["A2"]],Y)
    # backpropagation
    grads<-lin_backward_propagation(parameters,cache,X,Y)
    if (print_cost & i%%1000==0) print(cost)
    # parameter update
    parameters<-update_parameters(parameters,grads,learning_rate) 
    
  }
  return(parameters)
}



# predict
lin_predict<-function(parameters,X){
  cache<-lin_forward_propagation(X,parameters)
  predictions<-cache[["A2"]]
}

# parameters<-nn_model(X, Y, n_h=4, num_iterations = 10000, print_cost=F)
# predictions<-predict(parameters,X)
# prop.table(table(predictions==Y))

