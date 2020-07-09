#https://tensorflow.rstudio.com/tutorials/beginners/
#install.packages("tensorflow")
# My workaround:
#   I have copied the following files
# 
# libcrypto-1_1-x64.*
#   libssl-1_1-x64.*
#   from D:\Anaconda3\Library\bin to D:\Anaconda3\DLLs.

#https://github.com/conda/conda/issues/9746

#install.packages("reticulate")
#install.packages("keras")

# library(tensorflow)
# install_tensorflow()
#if(.Platform$OS.type == "windows") 
if (dir.exists("C:/Users/k_min")){
  Sys.setenv(PATH= paste("C:/Users/k_min/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
} else {
  Sys.setenv(PATH= paste("C:/Users/kmingoulin/Anaconda3/envs/r-tensorflow/Library/bin",Sys.getenv()["PATH"],sep=";"))
}
#
#

library(reticulate)
library(tensorflow)
#install_tensorflow()
# tf$constant("Hellow Tensorflow")


library(keras)


config <- py_config()
config$numpy


mnist <- dataset_mnist()

mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

summary(model)

model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

predictions <- predict_proba(model, mnist$test$x)
head(predictions, 2)

model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

model %>% 
  evaluate(mnist$train$x, mnist$train$y, verbose = 0)
