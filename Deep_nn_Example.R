# source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")

library(rhdf5)

# 1. load data
path_to_h5<-"C:/Kosta Work/Python/2020/datasets/train_catvnoncat.h5"
path_to_h5_test<-"C:/Kosta Work/Python/2020/datasets/test_catvnoncat.h5"
h5ls(path_to_h5)
h5ls(path_to_h5_test)

train_set_x_orig <- h5read(path_to_h5, "/train_set_x")
train_y <- h5read(path_to_h5, "/train_set_y")

test_set_x_orig <- h5read(path_to_h5_test, "/test_set_x")
test_y <- h5read(path_to_h5_test, "/test_set_y")

#tt<-as.vector(train_set_x_orig[,,,1])
train_x_flatten<-apply(train_set_x_orig,4,cbind)
dim(train_y)<-c(1,dim(train_x_flatten)[2])

test_x_flatten<-apply(test_set_x_orig,4,cbind)
dim(test_y)<-c(1,dim(test_x_flatten)[2])

# 2. standartize

train_x<-train_x_flatten/255L
test_x<-test_x_flatten/255L

library(openxlsx)
# train_x<-as.matrix(read.xlsx("Cat_x.xlsx",colNames = F,startRow = 2))
# train_y<-as.matrix(read.xlsx("Cat_y.xlsx",colNames = F,startRow = 2))
# 3. Model constants
#layers_dims<-c(12288, 20, 7, 5, 1)
layers_dims<-c(12288, 25, 10, 1)

# 4. Model: L-layer

L_layer_model<-function(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=F){
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
    cost<-compute_cost(AL, Y)
    
    # Backward propagation.
    grads<-L_model_backward(AL, Y, caches)
    
    # Update parameters.
    parameters<-update_parameters(parameters, grads, learning_rate)
    
    # print cost
    if(print_cost & (i %% 100) ==0){
      print(cost)
    }
  }
  return(parameters)
}


parameters<-L_layer_model(train_x, train_y, learning_rate = 0.0075, layers_dims, num_iterations = 2500, print_cost = T)

predict_nn_L_layer<-function(X, parameters){
  AL<-L_model_forward(X, parameters)[["AL"]]
}

AL_test<-predict_nn_L_layer(test_x, parameters)

table(test_y==(AL_test>.5))