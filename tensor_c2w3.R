if (dir.exists("C:/Users/k_min")){
  Sys.setenv(PATH= paste("C:/Users/k_min/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
} else {
  Sys.setenv(PATH= paste("C:/Users/kmingoulin/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
}


library(reticulate)
library(tensorflow)
library(keras)

library(rhdf5)

#load data
path_to_h5<-"C:/ProjectExchange/train_signs.h5"
path_to_h5_test<-"C:/ProjectExchange/test_signs.h5"
h5ls(path_to_h5)
h5ls(path_to_h5_test)

train_set_x_orig <- h5read(path_to_h5, "/train_set_x")
train_y <- h5read(path_to_h5, "/train_set_y")

test_set_x_orig <- h5read(path_to_h5_test, "/test_set_x")
test_y <- h5read(path_to_h5_test, "/test_set_y")

# scale
train_set_x_orig<-train_set_x_orig/255
test_set_x_orig<-test_set_x_orig/255

#train_x_flatten<-t(apply(train_set_x_orig,4,cbind))

 train_set_x_orig<-aperm(train_set_x_orig,c(4,1,2,3))
 test_set_x_orig<-aperm(test_set_x_orig,c(4,1,2,3))

model2 <- keras_model_sequential() %>%
  layer_flatten(data_format="channels_first",input_shape = c(3,64,64)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(6, activation = "softmax")

summary(model2)

model2 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


model2 %>% 
  fit(
    x = train_set_x_orig, y = train_y,
    learning_rate=0.0001,
    epochs = 50,
    validation_split = 0.2,
    verbose = 2
  )