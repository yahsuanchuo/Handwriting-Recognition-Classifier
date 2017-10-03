  #Load the binary data of MNIST dataset
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}
load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

#Load train and test data
train_data <- load_image_file("train-images.idx3-ubyte")
test_data <- load_image_file('t10k-images.idx3-ubyte')
#Load the label of train and test data
train_label <- load_label_file('train-labels.idx1-ubyte')
test_label <- load_label_file('t10k-labels.idx1-ubyte')

#library R function of mxnet, methods
library(mxnet)
library(methods)

train <- data.matrix(train_data[[2]])
test <- data.matrix(test_data[[2]])
  
train <- as.data.frame(train)
train_label <- as.data.frame(train_label)
train_all <- cbind(train_label,train)
table(train_label)

set.seed(101)
indexes <- sample(1:nrow(train_all), size=0.1*i*nrow(train_all))
train_part <- train_all[indexes,]

train_part_x <- train_part[,-1]
train_part_y <- train_part[,1]
train_part_x <- data.matrix(train_part_x)
train_part_y <- as.integer(train_part_y)


train.x <- t(train_part_x/255)
test <- t(test/255)

#Method 1 Configure the structure of the network
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=1024)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=512)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=256)


softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

#Choose CPU as the device
device.cpu <- mx.cpu()
#Train the neural network
mx.set.seed(0)
model1 <- mx.model.FeedForward.create(softmax, X=train.x, y=train_part_y,
                                                ctx=device.cpu, num.round=20, array.batch.size=100,
                                                learning.rate=0.1, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                                initializer=mx.init.uniform(0.1),
                                                epoch.end.callback=mx.callback.log.train.metric(100))
#Make the prediction 1
preds1 <- predict(model1, test)
pred.label1 <- max.col(t(preds1)) - 1
    
#Confusion matrix and accuracy
table(pred.label1,test_label)
correct1 <- sum(pred.label1==test_label)
accuracy1 <- correct1/ncol(test)
print(accuracy1)
    
FOM <- 0.06/2+(1-accuracy1)
print(FOM)



#Method 2: LeNet
# input
data <- mx.symbol.Variable('data')
# first conv

conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=100)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# first fullc 
flatten <- mx.symbol.Flatten(data=pool2)
fc2.1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc2.1, act_type="tanh")
# second fullc
fc2.2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)

lenet <- mx.symbol.SoftmaxOutput(data=fc2.2)

# Reshape the matrices into arrays
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

#Train the LeNet
device.cpu <- mx.cpu()
mx.set.seed(0)
model2 <- mx.model.FeedForward.create(lenet, X=train.array, y=train_part_y,
                                     ctx=device.cpu, num.round=10, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

#Make the prediction 2
preds2 <- predict(model2, test.array)
pred.label2 <- max.col(t(preds2)) - 1

#Confusion matrix and accuracy
table(pred.label2,test_label)
correct2 <- sum(pred.label2==test_label)
accuracy2 <- correct2/ncol(test)
print(accuracy2)

FOM= 0.1*i/2 + (1-accuracy2)
print(FOM)

rm(list=ls())
