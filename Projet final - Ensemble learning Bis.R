library(readr)
library(tidyverse)
library(mlbench)
library(caret)
library(caretEnsemble)

data <- read_csv("wdbc.data", 
                 col_names = c("ID number",
                               "Diagnosis",
                               "radius_mean",
                               "texture_mean",
                               "perimeter_mean",
                               "area_mean","smoothness_mean",
                               "compactness_mean",
                               "concavity_mean",
                               "concave_points_mean",
                               "symmetry_mean",
                               "fractal_dimension_mean",
                               "radius_SE","texture_SE",
                               "perimeter_SE","area_SE",
                               "smoothness_SE",
                               "compactness_SE",
                               "concavity_SE",
                               "concave_points_SE",
                               "symmetry_SE",
                               "fractal_dimension_SE",
                               "radius_worst",
                               "texture_worst",
                               "perimeter_worst",
                               "area_worst",
                               "smoothness_worst",
                               "compactness_worst",
                               "concavity_worst",
                               "concave_points_worst",
                               "symmetry_worst",
                               "fractal_dimension_worst"))

clean_data <- data %>% 
  select(c(contains("_mean"), Diagnosis)) %>% 
  drop_na()

el_dataset <- clean_data

# Example of Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# C5.0
set.seed(seed)
fit.c50 <- train(Diagnosis~., data=clean_data, method="C5.0", metric=metric, trControl=control)

# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Diagnosis~., data=clean_data, method="gbm", metric=metric, trControl=control, verbose=FALSE)

# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)

# Example of Bagging algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# Bagged CART
set.seed(seed)
fit.treebag <- train(Diagnosis~., data=clean_data, method="treebag", metric=metric, trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(Diagnosis~., data=clean_data, method="rf", metric=metric, trControl=control)

# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)

# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Diagnosis~., data=clean_data, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)