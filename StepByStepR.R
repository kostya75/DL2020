library(openxlsx)
W1<-unname(as.matrix(read.xlsx("W1.xlsx",colNames = F,startRow = 2)))
W2<-unname(as.matrix(read.xlsx("W2.xlsx",colNames = F,startRow = 2)))
W3<-unname(as.matrix(read.xlsx("W3.xlsx",colNames = F,startRow = 2)))
W4<-unname(as.matrix(read.xlsx("W4.xlsx",colNames = F,startRow = 2)))


b1<-matrix(0,nrow=20,ncol=1)
b2<-matrix(0,nrow=7,ncol=1)
b3<-matrix(0,nrow=5,ncol=1)
b4<-matrix(0,nrow=1,ncol=1)


parameters<-list(W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3,W4=W4,b4=b4)
source('C:/Kosta Work/R/DL2020/Deep_nn.R')

parameters$W4
X<-train_x
Y<-train_y
layers_dims<-c(12288, 20, 7, 5, 1)
learning_rate<-0.0075

parameters<-initialize_parameters_deep(layers_dims)
 ##################################
for(i in 1:5000){
# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
#AL, caches = L_model_forward(X, parameters)
AL_caches<-L_model_forward(X, parameters)
AL<-AL_caches[["AL"]]
caches<-AL_caches[["caches"]]


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
