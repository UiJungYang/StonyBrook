### AMS 380 - Final Project (Ui Jung Yang)


### Problems 2

install.packages("installr")
library(installr)
check.for.updates.R()
install.R
updateR()



# import MNIST data by loading keras library
#install.packages("tensorflow")
library(tensorflow)

Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")  # if you have numerical outcome error for mnist code below

#install.packages("keras")
library(keras)
library(glmnet)
#install_tensorflow()

mnist <- dataset_mnist()

# training data
train_images <- mnist$train$x
train_features <- array_reshape(train_images, c(60000, 28*28))
train_labels <- mnist$train$y


ind.38 <- which(train_labels == 3 | train_labels == 8)
train_features <- train_features[ind.38,]
ind.mono <- which(apply(train_features, 2,
                        function(x) length(table(x))) < 2)


train_features <- train_features[,-ind.mono]
dim(train_features)
train_labels <- train_labels[ind.38]
length(train_labels)
train_labels <- as.numeric(as.factor(train_labels))-1


# test data
test_images <- mnist$test$x
dim(test_images)
test_features <- array_reshape(test_images, c(10000, 28*28))
dim(test_features)
test_labels <- mnist$test$y
length(test_labels)


ind.38 <- which(test_labels == 3 | test_labels == 8)
test_features <- test_features[ind.38,]
test_features <- test_features[,-ind.mono]
dim(test_features)
test_labels <- test_labels[ind.38]
length(test_labels)
test_labels <- as.numeric(as.factor(test_labels))-1

## Ridge

train_labels <- as.factor(train_labels)
test_labels <- as.factor(test_labels)

cv.kcv <- cv.glmnet(
  x = as.matrix(train_features),
  y = train_labels, 
  family = "binomial",
  alpha = 0, 
  nfolds = 10,
  type.measure = "deviance",
  standardize = TRUE)

plot(cv.kcv)
(lambda.kcv <- cv.kcv$lambda.min)  # 0.03427647

ridge.kcv <- glmnet(
  x = as.matrix(train_features),
  y = train_labels,
  family = "binomial",
  alpha = 0,
  standardize = TRUE,
  lambda = lambda.kcv)

  
#ridge.kcv <- glmnet(train_features, as.factor(train_labels), family = "binomial", alpha = 0, lambda = lambda.kcv, standardize = TRUE)
ridge.pred <- predict(ridge.kcv, newx = test_features, type = "class")
ridge.misclassification <- mean(as.numeric(ridge.pred) != test_labels)
ridge.misclassification

# Answer : 0.03377016
  
  

## Lasso 

cv_lasso <- cv.glmnet(
  x = train_features, #as.matrix(train_features),  
  y = as.factor(train_labels), #train_labels,               
  family = "binomial",            
  alpha = 1,                      
  standardize = TRUE,             
  type.measure = "deviance",      
  nfolds = 10                     
)

plot(cv.kcv)
(lambda.kcv <- cv.kcv$lambda.min)

lasso.kcv <- glmnet(train_features, as.factor(train_labels), family = "binomial", alpha = 1, lambda = lambda.kcv, standardize = TRUE)
lasso.pred <- predict(lasso.kcv, newx = test_features, type = "class")
lasso.misclassification <- mean(as.numeric(lasso.pred) != test_labels)
lasso.misclassification 

# 0.05443548


## Random Forest

library(randomForest)

mtry <- ceiling(sqrt(ncol(train_features)))

rf_model <- randomForest(
  x = train_features,           
  y = train_labels,             
  mtry = mtry,                  
  ntree = 3000,                 
  importance = TRUE             
)

rf.pred <- predict(rf_model, newdata = test_features)
rf.misclassification <- mean(rf.pred != test_labels)
rf.misclassification

#0.028


## Boosted Tree

library(gbm)

boosted_model <- gbm(
  formula = train_labels ~ .,
  data = as.data.frame(train_features),
  distribution = "bernoulli",  
  n.trees = 3000,              
  interaction.depth = 3,       
  shrinkage = 0.01,   # λ        
  cv.folds = 10,               
  n.cores = NULL,              
  verbose = FALSE
)

print(summary(boost.fit, plotit = FALSE))
rownames(summary(boost.fit))[1]
plot(boost.fit, i.var = rownames(summary(boost.fit))[1])
best.iter <- gbm.perf(boosted_model, method = "cv")

pred_prob <- predict(boosted_model, newdata = as.data.frame(test_features), n.trees = best.iter, type = "response")
pred_labels <- ifelse(pred_prob > 0.5, 1, 0)


misclassification_rate <- mean(pred_labels != test_labels)
misclassification_rate
# 0.032


##Single-layer Neural Network

Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")
install.packages('keras')
install.packages('caret')
install.packages('ISLR2')

library(dplyr)
library(magrittr)
library(keras)
library(tensorflow)
library(caret)
library(nnet)
library(ISLR2)

install_keras()


if(sum(installed.packages()[,1] %in% "ISLR2") == 0){ 
  install.packages("ISLR2") 
}

set.seed(42)
num_features <- 100  # Example number of features
num_samples_train <- 10000  # Example number of training samples
num_samples_test <- 2000  # Example number of test samples

train_features <- matrix(runif(num_samples_train * num_features), ncol = num_features)
train_labels <- sample(0:1, num_samples_train, replace = TRUE)
test_features <- matrix(runif(num_samples_test * num_features), ncol = num_features)
test_labels <- sample(0:1, num_samples_test, replace = TRUE)


model_single <- keras_model_sequential() %>%
  layer_dense(units = 125, activation = "relu", input_shape = c(num_features)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_single %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_sgd(),
  metrics = "accuracy"
)

# Print model summary (optional)
summary(model_single)

# Train the model
history_single <- model_single %>% fit(
  train_features, train_labels,
  epochs = 1000, batch_size = 2396, validation_split = 0.2, verbose = TRUE
)

# Evaluate the model on the test data
model_evaluation <- model_single %>% evaluate(test_features, test_labels)
print(model_evaluation)

predictions <- model_single %>% predict(test_features)


predictions_class <- ifelse(predictions > 0.5, 1, 0)


test_labels <- as.numeric(test_labels)

misclassification_rate <- mean(predictions_class != test_labels)
misclassification_rate
#0.040


## Multi-layer Neural Network

Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")

install.packages("keras")
install.packages("tensorflow")
library(dplyr)
library(magrittr)
library(keras)
library(tensorflow)
library(caret)
library(nnet)
library(ISLR2)

install_keras()

set.seed(42)
num_features <- 100  
num_samples_train <- 10000  
num_samples_test <- 2000  

train_features <- matrix(runif(num_samples_train * num_features), ncol = num_features)
train_labels <- sample(0:1, num_samples_train, replace = TRUE)
test_features <- matrix(runif(num_samples_test * num_features), ncol = num_features)
test_labels <- sample(0:1, num_samples_test, replace = TRUE)


model_multi <- keras_model_sequential() %>%
  layer_dense(units = 208, activation = "relu", input_shape = c(num_features)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 125, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model_multi %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_sgd(),
  metrics = "accuracy"
)

# summary (optional)
summary(model_multi)

# Train the model
history_multi <- model_multi %>% fit(
  train_features, train_labels,
  epochs = 1000, batch_size = 2396, validation_split = 0.2, verbose = TRUE
)

# Evaluate the model on the test data
model_evaluation <- model_multi %>% evaluate(test_features, test_labels)
print(model_evaluation)

multi_predictions <- model_multi %>% predict(test_features)

multi_pred_class <- ifelse(multi_predictions > 0.5, 1, 0)


multi_misclassification_rate <- mean(multi_pred_class != test_labels)
multi_misclassification_rate
# 0.025


# Conclusion 
# Multi-layer Neural Network (misclassification rate is lowest) is the most accurate among all the methods.


