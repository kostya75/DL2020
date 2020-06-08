#lin testing

library(openxlsx)
library(data.table)
df<-read.xlsx("C:/ProjectExchange/Targeting.xlsx")
set.seed(4)
train_ind<-runif(nrow(df))<.7

df_train<-df[train_ind,]
df_test<-df[!train_ind,]



X<-t(as.matrix(df_train[,-1]))
Y_lin<-matrix(df_train[,1],ncol=nrow(df_train))


X_test<-t(as.matrix(df_test[,-1]))
Y_lin_test<-matrix(df_test[,1],ncol=nrow(df_test))


# parameters<-initialize_parameters(layer_sizes(X,Y_lin))
# 
# 
# cache<-lin_forward_propagation(X,parameters)
# cost<-lin_compute_cost(cache[["A2"]],Y_lin)
# print(cost)
# grads<-lin_backward_propagation(parameters,cache,X,Y_lin)
# parameters<-update_parameters(parameters,grads,learning_rate=1.2)


tt<-lin_nn_model(X, Y_lin, n_h=40, num_iterations = 50000, print_cost=T,learning_rate = .25)
