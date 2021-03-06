library(openxlsx)
library(data.table)
df<-read.xlsx("C:/ProjectExchange/Targeting.xlsx")
set.seed(4)
train_ind<-runif(nrow(df))<.9

df_train<-df[train_ind,]
df_test<-df[!train_ind,]

# X<-t(as.matrix(df_train[,-1]))
# Y<-matrix(df_train[,1]>0,ncol=nrow(df_train))
# 
# 
# X_test<-t(as.matrix(df_test[,-1]))
# Y_test<-matrix(df_test[,1]>0,ncol=nrow(df_test))
# 
# 
# parameters<-nn_model(X, Y, n_h=40, num_iterations = 20000, print_cost=T,learning_rate=.025)
# 
# cat("train\n")
# predictions<-predict(parameters,X)
# prop.table(table(predictions==Y))
# prop.table(table(predictions,Y))
# 
# cat("test\n")
# # model suffers from overfitting. need to regularize
# predictions_test<-predict(parameters,X_test)
# prop.table(table(predictions_test==Y_test))
# prop.table(table(predictions_test,Y_test))

################### nn3 ###############
# source("C:/Users/k_min/Documents/ML2020/nn3.R")
# XX<-t(X)
# YY<-t(rbind(Y*1,(!Y)*2))
# parameters_nn3<-nn3(XX,YY, hidden_layer_size=20, num_labels=2,lambda=1,method="BFGS")
# 
# table(predict_nn3(parameters_nn3, XX))
# table(predict_nn3(parameters_nn3, XX)==YY)

df_train_lr<-df_train
df_train_lr$Response<-(df_train_lr$Response>0)
optimal_train<-gdlreg2(Response~.,df_train_lr,theta=rep(0,510),lambda=0, method="BFGS" ,normalize = T)
predicted_lr<-predictLR(optimal_train,df_train_lr[,-1])

cat("train LR\n")
prop.table(table((predicted_lr>.5)==df_train_lr$Response))
prop.table(table((predicted_lr>.5),df_train_lr$Response))

cat("test LR\n")
df_test_lr<-df_test
df_test_lr$Response<-(df_test_lr$Response>0)
predicted_lr_test<-predictLR(optimal_train,df_test_lr[,-1])
prop.table(table((predicted_lr_test>.5)==df_test_lr$Response))
prop.table(table((predicted_lr_test>.5),df_test_lr$Response))


# #################################### L_layer
# layers_dims<-c(509,20,10,5,1)
# 
# X<-featureNormalize2(t(X),F)
# X<-X$x
# X<-t(X)
# parameters<-L_layer_model(X, Y, learning_rate = 0.01, layers_dims, num_iterations = 2500, print_cost = T)
