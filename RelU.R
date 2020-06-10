
RelU<-function(Z0,leaky){
  RelU_grad(Z0,leaky)*Z0
}

# tt2<-RelU(Y_lin_pred,0.1)
# table(tt2)

RelU_grad<-function(ZZ,leaky){
  # row_Z<-dim(Z)[1]
  # col_Z<-dim(Z)[2]
  #temp_compare<-(Z>=0)
  temp_compare<-structure(vapply(ZZ,
         function(x) if(x>=0) 1 else leaky
         ,numeric(1)),
         dim=dim(ZZ))
  #res<-matrix(mapply(max,Z,temp_compare),nrow=row_Z,ncol=col_Z)
}

# sapply(Z,function(x) { 
#   if(x>0){ 
#     res<-1
#     } else {
#       res<-leaky
#   }
#   res
#   }
# )



tt<-rel_nn_model(X, Y_lin, n_h=50, num_iterations = 5000, print_cost=T,learning_rate = .05)
par_rel<-rel_predict(parameters,X)
