library(openxlsx)
library(data.table)
df<-read.xlsx("C:/ProjectExchange/Targeting.xlsx")
set.seed(4)
train_ind<-runif(nrow(df))<.9

df_train<-df[train_ind,]
df_test<-df[!train_ind,]

X<-t(as.matrix(df_train[,-1]))
Y<-matrix(df_train[,1]>0,ncol=nrow(df_train))


X_test<-t(as.matrix(df_test[,-1]))
Y_test<-matrix(df_test[,1]>0,ncol=nrow(df_test))


source("TargetingFunctions.R")
source("Deep_nn_alternative2_Clean.R")

#parameters<-L_layer_model(train_x, train_y, learning_rate = 0.0075, layers_dims, num_iterations = 2500, print_cost = T)

layers_dims<-c(509,100,50,20,10,5,1)

X<-featureNormalize2(t(X),F)
X<-X$x
X<-t(X)
parameters<-L_layer_model(X, Y, learning_rate = 0.008, layers_dims, num_iterations = 10000, print_cost = T)

AL_test<-predict_nn_L_layer(X, parameters)

table(Y==(AL_test>.5))
prop.table(table(Y,(AL_test>.5)))
