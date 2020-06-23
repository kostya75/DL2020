library(openxlsx)
library(data.table)
df<-read.xlsx("C:/ProjectExchange/Targeting.xlsx")
set.seed(4)
train_ind<-runif(nrow(df))<.9

df_train<-df[train_ind,]
#df_train<-rbindlist(list(df_train[1:8136,],df_train[8138:13141,]))
df_test<-df[!train_ind,]

X<-t(as.matrix(df_train[,-1]))
Y<-matrix(df_train[,1]>0,ncol=nrow(df_train))


X_test<-t(as.matrix(df_test[,-1]))
Y_test<-matrix(df_test[,1]>0,ncol=nrow(df_test))


source("TargetingFunctions.R")
source("Deep_nn_alternative2_Clean.R")

#parameters<-L_layer_model(train_x, train_y, learning_rate = 0.0075, layers_dims, num_iterations = 2500, print_cost = T)

layers_dims<-c(509,100,50,20,12,6,1)

X<-featureNormalize2(t(X),F)
scaling_test<-X$scalingMatrix
X<-X$x
X<-t(X)
parameters<-L_layer_model(X, Y, learning_rate = 0.02, layers_dims, num_iterations = 2000, print_cost = T)
#parameters_3136<-parameters
#parameters<-parameters_3136

AL_train<-predict_nn_L_layer(X, parameters)

prop.table(table(Y==(AL_train>.5)))
prop.table(table(Y,(AL_train>.5)))

# test set
#X_test<-t((t(X_test)-scaling_test[rep(1,ncol(X_test)),])/scaling_test[rep(2,ncol(X_test)),])
# mus<-scalingMatrix[rep(1,nrow(X)),]
# sds<-scalingMatrix[rep(2,nrow(X)),]


AL_test<-predict_nn_L_layer(X_test, parameters)

prop.table(table(Y_test==(AL_test>.5)))
prop.table(table(Y_test,(AL_test>.5)))


######################### with REGULArizatio
parameters<-L_layer_model_with_regularization(X, Y, learning_rate = 0.02, layers_dims, num_iterations = 3000, print_cost = T,lambd = .1)
#parameters_3136<-parameters
#parameters<-parameters_3136

AL_train<-predict_nn_L_layer(X, parameters)

prop.table(table(Y==(AL_train>.5)))
prop.table(table(Y,(AL_train>.5)))

# test set
#X_test<-t((t(X_test)-scaling_test[rep(1,ncol(X_test)),])/scaling_test[rep(2,ncol(X_test)),])
# mus<-scalingMatrix[rep(1,nrow(X)),]
# sds<-scalingMatrix[rep(2,nrow(X)),]


AL_test<-predict_nn_L_layer(X_test, parameters)

prop.table(table(Y_test==(AL_test>.5)))
prop.table(table(Y_test,(AL_test>.5)))
