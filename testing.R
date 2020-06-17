
A1<-relu(W1%*%X+b1)$A
A2<-relu(W2%*%A1+b2)$A
AL<-sigmoid(W3%*%A2+b3)$A




b1<-b1[,rep(1,times=4)]
b2<-b2[,rep(1,times=4)]
b3<-b3[,rep(1,times=4)]

A1<-relu(W1%*%X+b1)$A
A2<-relu(W2%*%A1+b2)$A
AL<-sigmoid(W3%*%A2+b3)$A