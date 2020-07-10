if (dir.exists("C:/Users/k_min")){
  Sys.setenv(PATH= paste("C:/Users/k_min/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
} else {
  Sys.setenv(PATH= paste("C:/Users/kmingoulin/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
}


library(reticulate)
library(tensorflow)
library(keras)



model3 <- keras_model_sequential() %>%
  #layer_flatten(input_shape = c(3,64,64)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 24, activation = "relu") %>%
  layer_dropout(0.2) %>%
  # layer_dense(units = 64, activation = "relu") %>%
  # layer_dropout(0.2) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(1, activation = "sigmoid")

#summary(model3)

model3 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


model3 %>% 
  fit(
    x = as.matrix(df_train_lr[,-1]), y = df_train_lr[,1],
    epochs = 4000,
    learning_rate=.0001,
    validation_split = 0.2,
    batch_size=64,
    verbose = 1
  )

# predictions <- predict_class(model3, df_test_lr[,-1]))
# head(predictions, 2)

model3 %>% 
  evaluate(as.matrix(df_train_lr[,-1]), df_train_lr[,1], verbose = 0)

model3 %>% 
  evaluate(as.matrix(df_test_lr[,-1]), df_test_lr[,1], verbose = 0)

pred_nn_train<-predict_proba(model3, as.matrix(df_train_lr[,-1]))
pred_nn_test<-predict_proba(model3, as.matrix(df_test_lr[,-1]))

prop.table(table(df_train_lr[,1],pred_nn_train>.5))
prop.table(table(df_test_lr[,1],pred_nn_test>.5))

K <- backend()
K$clear_session()

