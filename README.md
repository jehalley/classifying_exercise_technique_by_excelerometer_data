---
title: "Machine Learning Course Project"
author: "Jeff Halley"
date: "April 7, 2019"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction
In  physical exercise, technique is of primary importance for injury prevention and for producing the most benefit for the exerciser. Recent advances in wearable motion tracking devices have made it possible to take simple measurements of movement using multiple-axis acceleromters. In their 2013 paper, H. Fuks et al. designed a study aimed at using accelerometer measurements to classify an exercisers technique as either correct or incorrect in one of four different ways. Accelerometer measurements were taken on 6 participants that performed 10 repetitions of the exercise in 5 different variations of the exercise. The first variation wered performed using correct technique, and the other variations were peformed with technique that was flawed in a particular way (for example, lifting the dumbbell only half way).

This project describes the creation of a model using Fuks et. al's data that can be used to classify a barbell curl as either performed using correct technique or incorrect technique based on acceloremter data taken during the exercise. The model described in this project was created using the random forest method in the Caret package. The model was trained on 14718 observations from Fuk et. al's experiments and predicted 4904 test cases with 99% accuracy. The model's accuracy was assessed using k-fold cross validation(k=5) and was found to be 99% accurate in all 5 tests.

## Downloading and Preprocessing Data
```{r}
setwd("/Users/JeffHalley/coursera/machinelearning")
pml.training<-read.csv("pml-training.csv")
pml.test<-read.csv("pml-testing.csv")
```

The traing data contain 160 different variables, but 100 of these variables have no data recorded for most observations. Early attempts at training a model on this raw data were unsuccessful because of the large number of missing values. Besides removing the variables that were mostly missing from the data set I also removed some variables that may have led to accurate predictions but that were not based solely on accelerometer readings. For instance, I removed the name of participant variable because it was not directly relevant to the accelerometer readings  and I removed the  time-stamp variable and the related "new window" variable because the different variations of the exercise seemed to be performed sequentially, and thus if these were not excluded a repetition may be correctly classified based on when it was performed rather than how it was performed. The following code creates two new data frames that do not contain the unwanted variables.

```{r}
data<-pml.training[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
test.data<-pml.test[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
```

Before training a model, I need to separate the training data set into a subset that I can use for validation. The following code uses the Caret package's create data partition to randomly subset the pml-training data into a training set and testing set 

```{r}
library(caret)
inTrain = createDataPartition(data$classe, p = 3/4)[[1]]
training = data[ inTrain,]
testing = data[-inTrain,]
```

##Creating the Model
I used Caret's random forest method to train the model because I was concerned about overfitting and I was also interested in identifying the variables that were most important to classifying the repetitions. 

The classification of each lift's technique is recorded in the "classe" variable and since I removed unwanted variables from the training and test data sets I trained the model to predict classe based on all other variables. 

Initial attempts at training the model were extremely time consuming. To reduce computation time I followed the parallel-processing method described in https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md.


```{r}
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

#set the fitControl settings to include 5 sets of cross validation and to use parallell processing. 
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
#perform a random forest model fit using the fit control settings.
modfit.rf<-train(classe~.,data=training,method="rf",na.action="na.omit",trControl = fitControl)

#stop parallell processing
stopCluster(cluster)
registerDoSEQ()
```

To determine the relative importance of each variable to the model I used Caret's varimp function. 

```{r}
varImp(modfit.rf)
```

These results reveal that the "roll_belt" variable are the most important for predicting correct dumbbell curl technique.

##Assessing the model's accuracy
To assess the model's accuracy I used it to predict the classe variable on the "testing" validation data set that I created earlier and then I used confusionMatrix to assess it's accuracy.

```{r}
prediction.rf <- predict(modfit.rf, newdata =testing)
#assess accuracy of model
confusionMatrix(prediction.rf, testing$classe)$overall['Accuracy']
```

Thus, the model accurately predicted 99% of the 4904 observations contained in the validation set. 

##Cross Validation to assess out of sample error rate
To further assess the model's accuracy I used k-fold cross validation using a method described at http://t-redactyl.io/blog/2015/10/using-k-fold-cross-validation-to-estimate-out-of-sample-accuracy.html
```{r}
k.folds <- function(k) {
  folds <- createFolds(training$classe, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    model <- train(classe ~ ., 
                   data = training[folds[[i]],], method = "rf", na.action="na.omit",trControl = fitControl)
    predictions <- predict(object = model, newdata = training[-folds[[i]],])
    accuracies.dt <- c(accuracies.dt, 
                       confusionMatrix(predictions, training[-folds[[i]], ]$classe)$overall[[1]])
  }
  accuracies.dt
}

set.seed(567)
accuracies.dt <- c()
accuracies.dt <- k.folds(5)
accuracies.dt
```

In all 5 of these tests the model also exhibits 99% accuracy thus this model is suitable for classifying dumbbell curl technique from accelerometer readings with an expected out of sample error rate of 1%.
