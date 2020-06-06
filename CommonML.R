#######################################
###### align with ML2020 project ######


# common functions

# sigmoid function
sigmoid<-function(z){
  g<-1/(1+exp(-z))
}


# sigmoid gradient

sigmoidGradient<-function(z){
  g<-sigmoid(z)*(1-sigmoid(z))
}

# feature normalize

featureNormalize<-function(xm, infl){
  
  scalingMatrix<-apply(xm,2,function(x){
    cbind(mean(x),sd(x))
  })
  rownames(scalingMatrix)<-c("mu","sigma")
  
  # if model has constant, do not scale the constant
  if(infl==1){
    scalingMatrix[1,1]<-0
    scalingMatrix[2,1]<-1
  }
  
  # Alternative
  # X_norm<-
  #  matrix(
  #    mapply(function(x,m,s){(x-m)/s},x=as.vector(t(xm)),m=scalingMatrix[1,],s=scalingMatrix[2,]),
  #    nrow=dim(xm)[1],byrow=T)
  
  mus<-scalingMatrix[rep(1,nrow(xm)),]
  sds<-scalingMatrix[rep(2,nrow(xm)),]
  X_norm<-(xm-mus)/sds
  out<-list(x=X_norm,scalingMatrix=scalingMatrix)
}

# pass matrix size to create empty matrix of same size with a value
ones_zeros<-function(value,size){
  if(length(size)!=2  | !is.numeric(size)) stop("Size should have length of 2 and be numeric")
  matrix(value,nrow=size[1],ncol=size[2])
}
