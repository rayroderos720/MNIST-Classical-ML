#Ray Anthony Roderos, Nischitha Rao, Qimeng Deng
#Machine Learning Midterm Project - Handwriting Analysis
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Table of Contents

#Line 20 - 84 Loading of Files and Checking of the Data including plotting the first four observation
#Line 84 - 719 Data Engineering - Adding of variables to the test and Training Set
#Line 740 - Multiple Linear Regression Model
#Line 794 - Ridge Regression
#Line 820 - LASSO
#Line 938 - Support Vector Machines (SVM)
#Line 1023 - Decision Trees
#Line 1065 - Random Forest
#Line 1099 - K-Means
#Line 1171 - Saving the environment




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


#Step 1 - Load the Project Files and get the details of the datasets

getwd()
MNIST_Test <- read.csv("MNIST - TEST.csv", header=TRUE)
MNIST_Train <-read.csv("MNIST - TRAIN.csv", header=TRUE)

#To summarize

head(MNIST_Test)
head(MNIST_Train)

#To Visualize


rotate <- function(x){ #creates a function to rotate the image
  t(apply(x, 2, rev))
} 

#creates a 28x28 matrix from the data of the 1st to the 4th rows, makes the elements numeric
#and applies the rotate function

image1_m <- matrix((MNIST_Train[1,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE) 
image1_m <- apply(image1_m, 2, as.numeric)
image1_m <- rotate(image1_m)

image2_m <- matrix((MNIST_Train[3,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image2_m <- apply(image2_m, 2, as.numeric)
image2_m <- rotate(image2_m)

image3_m <- matrix((MNIST_Train[4,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image3_m <- apply(image3_m, 2, as.numeric)
image3_m <- rotate(image3_m)

image4_m <- matrix((MNIST_Train[5,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image4_m <- apply(image4_m, 2, as.numeric)
image4_m <- rotate(image4_m)

#create a function to plot the matrix

plot_number <- function(x){
  x <- apply(x, 2, as.numeric)
  image(1:28, 1:28, x, col=gray((0:255)/255))
}

graphics.off() #erases all plots
par(mfrow=c(2,2),pty="s") #creates a 2x2 slots in the plot, pty="s" makees the plot square

plot_number(image1_m) #plots the 1st image (5)
plot_number(image2_m) #plots the 2nd image (4)
plot_number(image3_m) #plots the 3rd image (1)
plot_number(image4_m) #plots the 4th image (9)


#For MNIST_Test and MNIST_Train, there are 10k and 60k oberservations 
#respectively with 784 variables in the form of pixels with values
#ranging in an 8-bit grayscale (0-255 from no black to the darkest black)
#It will create a visual representation of a number if made into a 28x28 matrix
#the reasearchers have determined that it is necesasry to add more variables
#to be able to extract more descriptions about the dataset



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Step 2 - Adding Variables to the Test_Set

#Make a new variable "Total_Pixels" - adding all values
MNIST_Colnames <- colnames(MNIST_Test)
MNIST_Colnames<-MNIST_Colnames[-1]

MNIST_Test$Total_Pixels <- rowSums(MNIST_Test[,MNIST_Colnames], na.rm=TRUE)

#Make new variables "Upper Half" and "Lower Half" - adding 
#values at the upper and lower half of the 28x28 matrix

MNIST_Colnames_Upperhalf <- MNIST_Colnames[1:392]
MNIST_Colnames_Lowerhalf <- MNIST_Colnames[393:784]
 
MNIST_Test$Lower_Half <- rowSums(MNIST_Test[,MNIST_Colnames_Lowerhalf], na.rm=TRUE)
MNIST_Test$Upper_Half <- rowSums(MNIST_Test[,MNIST_Colnames_Upperhalf], na.rm=TRUE)

#Make new variables "Left Half" and "Right Half" - adding 
#the values of the left an right half of the 28x28 matrix

LeftHalf_Matrix <- matrix(,nrow=28,ncol=14)
RightHalf_Matrix <- matrix(,nrow=28,ncol=14)

LeftHalf_Matrix[1,] <- c(1:14)
RightHalf_Matrix[1,] <- c(15:28)

for (i in 2:28){
  LeftHalf_Matrix[i,] <- LeftHalf_Matrix[i-1,]+28
  RightHalf_Matrix[i,] <- RightHalf_Matrix[i-1,]+28
}

LeftHalf_Vector <- as.vector(t(LeftHalf_Matrix))+1
RightHalf_Vector <- as.vector(t(RightHalf_Matrix))+1


MNIST_Test$Left_Half <- rowSums(MNIST_Test[LeftHalf_Vector], na.rm=TRUE)
MNIST_Test$Right_Half <- rowSums(MNIST_Test[RightHalf_Vector], na.rm=TRUE)


#Make new variables Quadrant1-4-adding
#the values of the 4 quandrants of the 28x28 matrix

Q1Matrix <- matrix(,nrow=14,ncol=14)
Q2Matrix <- matrix(,nrow=14,ncol=14)
Q3Matrix <- matrix(,nrow=14,ncol=14)
Q4Matrix <- matrix(,nrow=14,ncol=14)


Q1Matrix[1,] <- c(1:14)
Q2Matrix[1,] <- c(15:28)
Q3Matrix[1,] <- c(393:406)
Q4Matrix[1,] <- c(407:420)


for (i in 2:14){
  Q1Matrix[i,] <- Q1Matrix[i-1,]+28
  Q2Matrix[i,] <- Q2Matrix[i-1,]+28
  Q3Matrix[i,] <- Q3Matrix[i-1,]+28
  Q4Matrix[i,] <- Q4Matrix[i-1,]+28
}

Q1_Vector <- as.vector(t(Q1Matrix))+1
Q2_Vector <- as.vector(t(Q2Matrix))+1
Q3_Vector <- as.vector(t(Q3Matrix))+1
Q4_Vector <- as.vector(t(Q4Matrix))+1


MNIST_Test$Q1 <- rowSums(MNIST_Test[Q1_Vector], na.rm=TRUE)
MNIST_Test$Q2 <- rowSums(MNIST_Test[Q2_Vector], na.rm=TRUE)
MNIST_Test$Q3 <- rowSums(MNIST_Test[Q3_Vector], na.rm=TRUE)
MNIST_Test$Q4 <- rowSums(MNIST_Test[Q4_Vector], na.rm=TRUE)


##Make new variables QuadrantNQuartileN (in the notatation QNQn) -adding
#the values of the 16 quartiles of the 28x28 matrix

#Quadrant 1


Quadrant1Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile4_M <- matrix(,nrow=7,ncol=7)


Quadrant1Quartile1_M[1,] <- c(1:7)
Quadrant1Quartile2_M[1,] <- c(8:14)
Quadrant1Quartile3_M[1,] <- c(197:203)
Quadrant1Quartile4_M[1,] <- c(204:210)

for(i in 2:7){
  Quadrant1Quartile1_M[i,] <- Quadrant1Quartile1_M[i-1,]+28
  Quadrant1Quartile2_M[i,] <- Quadrant1Quartile2_M[i-1,]+28
  Quadrant1Quartile3_M[i,] <- Quadrant1Quartile3_M[i-1,]+28
  Quadrant1Quartile4_M[i,] <- Quadrant1Quartile4_M[i-1,]+28
}
  
Quadrant1Quartile1_V <-as.vector(t(Quadrant1Quartile1_M))+1
Quadrant1Quartile2_V <-as.vector(t(Quadrant1Quartile2_M))+1
Quadrant1Quartile3_V <-as.vector(t(Quadrant1Quartile3_M))+1
Quadrant1Quartile4_V <-as.vector(t(Quadrant1Quartile4_M))+1

Quadrant1Quartile4_V

MNIST_Test$Q1Q1 <- rowSums(MNIST_Test[Quadrant1Quartile1_V], na.rm=TRUE)
MNIST_Test$Q1Q2 <- rowSums(MNIST_Test[Quadrant1Quartile2_V], na.rm=TRUE)
MNIST_Test$Q1Q3 <- rowSums(MNIST_Test[Quadrant1Quartile3_V], na.rm=TRUE)
MNIST_Test$Q1Q4 <- rowSums(MNIST_Test[Quadrant1Quartile4_V], na.rm=TRUE)

#Quadrant 2


Quadrant2Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile4_M <- matrix(,nrow=7,ncol=7)


Quadrant2Quartile1_M[1,] <- c(15:21)
Quadrant2Quartile2_M[1,] <- c(22:28)
Quadrant2Quartile3_M[1,] <- c(211:217)
Quadrant2Quartile4_M[1,] <- c(218:224)

for(i in 2:7){
  Quadrant2Quartile1_M[i,] <- Quadrant2Quartile1_M[i-1,]+28
  Quadrant2Quartile2_M[i,] <- Quadrant2Quartile2_M[i-1,]+28
  Quadrant2Quartile3_M[i,] <- Quadrant2Quartile3_M[i-1,]+28
  Quadrant2Quartile4_M[i,] <- Quadrant2Quartile4_M[i-1,]+28
}

Quadrant2Quartile1_V <-as.vector(t(Quadrant2Quartile1_M))+1
Quadrant2Quartile2_V <-as.vector(t(Quadrant2Quartile2_M))+1
Quadrant2Quartile3_V <-as.vector(t(Quadrant2Quartile3_M))+1
Quadrant2Quartile4_V <-as.vector(t(Quadrant2Quartile4_M))+1

MNIST_Test$Q2Q1 <- rowSums(MNIST_Test[Quadrant1Quartile1_V], na.rm=TRUE)
MNIST_Test$Q2Q2 <- rowSums(MNIST_Test[Quadrant1Quartile2_V], na.rm=TRUE)
MNIST_Test$Q2Q3 <- rowSums(MNIST_Test[Quadrant1Quartile3_V], na.rm=TRUE)
MNIST_Test$Q2Q4 <- rowSums(MNIST_Test[Quadrant1Quartile4_V], na.rm=TRUE)

#Quadrant 3


Quadrant3Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile4_M <- matrix(,nrow=7,ncol=7)

Quadrant3Quartile1_M[1,] <- Q3Matrix[1,1:7]
Quadrant3Quartile2_M[1,] <- Q3Matrix[1,8:14]
Quadrant3Quartile3_M[1,] <- Q3Matrix[8,1:7]
Quadrant3Quartile4_M[1,] <- Q3Matrix[8,8:14]

for(i in 2:7){
  Quadrant3Quartile1_M[i,] <- Quadrant3Quartile1_M[i-1,]+28
  Quadrant3Quartile2_M[i,] <- Quadrant3Quartile2_M[i-1,]+28
  Quadrant3Quartile3_M[i,] <- Quadrant3Quartile3_M[i-1,]+28
  Quadrant3Quartile4_M[i,] <- Quadrant3Quartile4_M[i-1,]+28
}

Quadrant3Quartile1_V <-as.vector(t(Quadrant3Quartile1_M))+1
Quadrant3Quartile2_V <-as.vector(t(Quadrant3Quartile2_M))+1
Quadrant3Quartile3_V <-as.vector(t(Quadrant3Quartile3_M))+1
Quadrant3Quartile4_V <-as.vector(t(Quadrant3Quartile4_M))+1

MNIST_Test$Q3Q1 <- rowSums(MNIST_Test[Quadrant3Quartile1_V], na.rm=TRUE)
MNIST_Test$Q3Q2 <- rowSums(MNIST_Test[Quadrant3Quartile2_V], na.rm=TRUE)
MNIST_Test$Q3Q3 <- rowSums(MNIST_Test[Quadrant3Quartile3_V], na.rm=TRUE)
MNIST_Test$Q3Q4 <- rowSums(MNIST_Test[Quadrant3Quartile4_V], na.rm=TRUE)

#Quadrant 4

Quadrant4Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile4_M <- matrix(,nrow=7,ncol=7)

Quadrant4Quartile1_M[1,] <- Q4Matrix[1,1:7]
Quadrant4Quartile2_M[1,] <- Q4Matrix[1,8:14]
Quadrant4Quartile3_M[1,] <- Q4Matrix[8,1:7]
Quadrant4Quartile4_M[1,] <- Q4Matrix[8,8:14]

for(i in 2:7){
  Quadrant4Quartile1_M[i,] <- Quadrant4Quartile1_M[i-1,]+28
  Quadrant4Quartile2_M[i,] <- Quadrant4Quartile2_M[i-1,]+28
  Quadrant4Quartile3_M[i,] <- Quadrant4Quartile3_M[i-1,]+28
  Quadrant4Quartile4_M[i,] <- Quadrant4Quartile4_M[i-1,]+28
}

Quadrant4Quartile1_V <-as.vector(t(Quadrant4Quartile1_M))+1
Quadrant4Quartile2_V <-as.vector(t(Quadrant4Quartile2_M))+1
Quadrant4Quartile3_V <-as.vector(t(Quadrant4Quartile3_M))+1
Quadrant4Quartile4_V <-as.vector(t(Quadrant4Quartile4_M))+1

MNIST_Test$Q4Q1 <- rowSums(MNIST_Test[Quadrant4Quartile1_V], na.rm=TRUE)
MNIST_Test$Q4Q2 <- rowSums(MNIST_Test[Quadrant4Quartile2_V], na.rm=TRUE)
MNIST_Test$Q4Q3 <- rowSums(MNIST_Test[Quadrant4Quartile3_V], na.rm=TRUE)
MNIST_Test$Q4Q4 <- rowSums(MNIST_Test[Quadrant4Quartile4_V], na.rm=TRUE)

##Make new variables PartN -adding
#the values of the 49 parts with the x and y axis
#equally divided by 7 in the 28x28 matrix

Subset_df <- matrix(,nrow=49,ncol=16)

Subset_df[1,] <- c(1:4,29:32,57:60,85:88)
Subset_df

for (i in 2:7){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[8,] <- Subset_df[1,]+112

for (i in 9:14){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[15,] <- Subset_df[8,]+112

for (i in 16:21){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[22,] <- Subset_df[15,]+112

for (i in 23:28){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[29,] <- Subset_df[22,]+112

for (i in 30:35){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[36,] <- Subset_df[29,]+112

for (i in 37:42){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[43,] <- Subset_df[36,] +112

for (i in 44:49){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df <- Subset_df+1

MNIST_Test$Subset1 <- rowSums(MNIST_Test[Subset_df[1,]], na.rm=TRUE)
MNIST_Test$Subset2 <- rowSums(MNIST_Test[Subset_df[2,]], na.rm=TRUE)
MNIST_Test$Subset3 <- rowSums(MNIST_Test[Subset_df[3,]], na.rm=TRUE)
MNIST_Test$Subset4 <- rowSums(MNIST_Test[Subset_df[4,]], na.rm=TRUE)
MNIST_Test$Subset5 <- rowSums(MNIST_Test[Subset_df[5,]], na.rm=TRUE)
MNIST_Test$Subset6 <- rowSums(MNIST_Test[Subset_df[6,]], na.rm=TRUE)
MNIST_Test$Subset7 <- rowSums(MNIST_Test[Subset_df[7,]], na.rm=TRUE)
MNIST_Test$Subset8 <- rowSums(MNIST_Test[Subset_df[8,]], na.rm=TRUE)
MNIST_Test$Subset9 <- rowSums(MNIST_Test[Subset_df[9,]], na.rm=TRUE)
MNIST_Test$Subset10 <- rowSums(MNIST_Test[Subset_df[10,]], na.rm=TRUE)
MNIST_Test$Subset11 <- rowSums(MNIST_Test[Subset_df[11,]], na.rm=TRUE)
MNIST_Test$Subset12 <- rowSums(MNIST_Test[Subset_df[12,]], na.rm=TRUE)
MNIST_Test$Subset13 <- rowSums(MNIST_Test[Subset_df[13,]], na.rm=TRUE)
MNIST_Test$Subset14 <- rowSums(MNIST_Test[Subset_df[14,]], na.rm=TRUE)
MNIST_Test$Subset15 <- rowSums(MNIST_Test[Subset_df[15,]], na.rm=TRUE)
MNIST_Test$Subset16 <- rowSums(MNIST_Test[Subset_df[16,]], na.rm=TRUE)
MNIST_Test$Subset17 <- rowSums(MNIST_Test[Subset_df[17,]], na.rm=TRUE)
MNIST_Test$Subset18 <- rowSums(MNIST_Test[Subset_df[18,]], na.rm=TRUE)
MNIST_Test$Subset19 <- rowSums(MNIST_Test[Subset_df[19,]], na.rm=TRUE)
MNIST_Test$Subset20 <- rowSums(MNIST_Test[Subset_df[20,]], na.rm=TRUE)
MNIST_Test$Subset21 <- rowSums(MNIST_Test[Subset_df[21,]], na.rm=TRUE)

MNIST_Test$Subset22 <- rowSums(MNIST_Test[Subset_df[22,]], na.rm=TRUE)
MNIST_Test$Subset23 <- rowSums(MNIST_Test[Subset_df[23,]], na.rm=TRUE)
MNIST_Test$Subset24 <- rowSums(MNIST_Test[Subset_df[24,]], na.rm=TRUE)
MNIST_Test$Subset25 <- rowSums(MNIST_Test[Subset_df[25,]], na.rm=TRUE)
MNIST_Test$Subset26 <- rowSums(MNIST_Test[Subset_df[26,]], na.rm=TRUE)
MNIST_Test$Subset27 <- rowSums(MNIST_Test[Subset_df[27,]], na.rm=TRUE)
MNIST_Test$Subset28 <- rowSums(MNIST_Test[Subset_df[28,]], na.rm=TRUE)
MNIST_Test$Subset29 <- rowSums(MNIST_Test[Subset_df[29,]], na.rm=TRUE)
MNIST_Test$Subset30 <- rowSums(MNIST_Test[Subset_df[30,]], na.rm=TRUE)
MNIST_Test$Subset31 <- rowSums(MNIST_Test[Subset_df[31,]], na.rm=TRUE)
MNIST_Test$Subset32 <- rowSums(MNIST_Test[Subset_df[32,]], na.rm=TRUE)
MNIST_Test$Subset33 <- rowSums(MNIST_Test[Subset_df[33,]], na.rm=TRUE)
MNIST_Test$Subset34 <- rowSums(MNIST_Test[Subset_df[34,]], na.rm=TRUE)
MNIST_Test$Subset35 <- rowSums(MNIST_Test[Subset_df[35,]], na.rm=TRUE)
MNIST_Test$Subset36 <- rowSums(MNIST_Test[Subset_df[36,]], na.rm=TRUE)
MNIST_Test$Subset37 <- rowSums(MNIST_Test[Subset_df[37,]], na.rm=TRUE)
MNIST_Test$Subset38 <- rowSums(MNIST_Test[Subset_df[38,]], na.rm=TRUE)
MNIST_Test$Subset39 <- rowSums(MNIST_Test[Subset_df[39,]], na.rm=TRUE)
MNIST_Test$Subset40 <- rowSums(MNIST_Test[Subset_df[40,]], na.rm=TRUE)
MNIST_Test$Subset41 <- rowSums(MNIST_Test[Subset_df[41,]], na.rm=TRUE)
MNIST_Test$Subset42 <- rowSums(MNIST_Test[Subset_df[42,]], na.rm=TRUE)

MNIST_Test$Subset43 <- rowSums(MNIST_Test[Subset_df[43,]], na.rm=TRUE)
MNIST_Test$Subset44 <- rowSums(MNIST_Test[Subset_df[44,]], na.rm=TRUE)
MNIST_Test$Subset45 <- rowSums(MNIST_Test[Subset_df[45,]], na.rm=TRUE)
MNIST_Test$Subset46 <- rowSums(MNIST_Test[Subset_df[46,]], na.rm=TRUE)
MNIST_Test$Subset47 <- rowSums(MNIST_Test[Subset_df[47,]], na.rm=TRUE)
MNIST_Test$Subset48 <- rowSums(MNIST_Test[Subset_df[48,]], na.rm=TRUE)
MNIST_Test$Subset49 <- rowSums(MNIST_Test[Subset_df[49,]], na.rm=TRUE)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Adding Variables to the Training Set

#Make a new variable "Total_Pixels" - adding all values
MNIST_Colnames <- colnames(MNIST_Train)
MNIST_Colnames<-MNIST_Colnames[-1]

MNIST_Train$Total_Pixels <- rowSums(MNIST_Train[,MNIST_Colnames], na.rm=TRUE)

#Make new variables "Upper Half" and "Lower Half" - adding 
#values at the upper and lower half of the 28x28 matrix

MNIST_Colnames_Upperhalf <- MNIST_Colnames[1:392]
MNIST_Colnames_Lowerhalf <- MNIST_Colnames[393:784]

MNIST_Train$Lower_Half <- rowSums(MNIST_Train[,MNIST_Colnames_Lowerhalf], na.rm=TRUE)
MNIST_Train$Upper_Half <- rowSums(MNIST_Train[,MNIST_Colnames_Upperhalf], na.rm=TRUE)

#Make new variables "Left Half" and "Right Half" - adding 
#the values of the left an right half of the 28x28 matrix

LeftHalf_Matrix <- matrix(,nrow=28,ncol=14)
RightHalf_Matrix <- matrix(,nrow=28,ncol=14)

LeftHalf_Matrix[1,] <- c(1:14)
RightHalf_Matrix[1,] <- c(15:28)

for (i in 2:28){
  LeftHalf_Matrix[i,] <- LeftHalf_Matrix[i-1,]+28
  RightHalf_Matrix[i,] <- RightHalf_Matrix[i-1,]+28
}

LeftHalf_Vector <- as.vector(t(LeftHalf_Matrix))+1
RightHalf_Vector <- as.vector(t(RightHalf_Matrix))+1


MNIST_Train$Left_Half <- rowSums(MNIST_Train[LeftHalf_Vector], na.rm=TRUE)
MNIST_Train$Right_Half <- rowSums(MNIST_Train[RightHalf_Vector], na.rm=TRUE)


#Make new variables Quadrant1-4-adding
#the values of the 4 quandrants of the 28x28 matrix

Q1Matrix <- matrix(,nrow=14,ncol=14)
Q2Matrix <- matrix(,nrow=14,ncol=14)
Q3Matrix <- matrix(,nrow=14,ncol=14)
Q4Matrix <- matrix(,nrow=14,ncol=14)


Q1Matrix[1,] <- c(1:14)
Q2Matrix[1,] <- c(15:28)
Q3Matrix[1,] <- c(393:406)
Q4Matrix[1,] <- c(407:420)


for (i in 2:14){
  Q1Matrix[i,] <- Q1Matrix[i-1,]+28
  Q2Matrix[i,] <- Q2Matrix[i-1,]+28
  Q3Matrix[i,] <- Q3Matrix[i-1,]+28
  Q4Matrix[i,] <- Q4Matrix[i-1,]+28
}

Q1_Vector <- as.vector(t(Q1Matrix))+1
Q2_Vector <- as.vector(t(Q2Matrix))+1
Q3_Vector <- as.vector(t(Q3Matrix))+1
Q4_Vector <- as.vector(t(Q4Matrix))+1


MNIST_Train$Q1 <- rowSums(MNIST_Train[Q1_Vector], na.rm=TRUE)
MNIST_Train$Q2 <- rowSums(MNIST_Train[Q2_Vector], na.rm=TRUE)
MNIST_Train$Q3 <- rowSums(MNIST_Train[Q3_Vector], na.rm=TRUE)
MNIST_Train$Q4 <- rowSums(MNIST_Train[Q4_Vector], na.rm=TRUE)


##Make new variables QuadrantNQuartileN (in the notatation QNQn) -adding
#the values of the 16 quartiles of the 28x28 matrix

#Quadrant 1


Quadrant1Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant1Quartile4_M <- matrix(,nrow=7,ncol=7)


Quadrant1Quartile1_M[1,] <- c(1:7)
Quadrant1Quartile2_M[1,] <- c(8:14)
Quadrant1Quartile3_M[1,] <- c(197:203)
Quadrant1Quartile4_M[1,] <- c(204:210)

for(i in 2:7){
  Quadrant1Quartile1_M[i,] <- Quadrant1Quartile1_M[i-1,]+28
  Quadrant1Quartile2_M[i,] <- Quadrant1Quartile2_M[i-1,]+28
  Quadrant1Quartile3_M[i,] <- Quadrant1Quartile3_M[i-1,]+28
  Quadrant1Quartile4_M[i,] <- Quadrant1Quartile4_M[i-1,]+28
}

Quadrant1Quartile1_V <-as.vector(t(Quadrant1Quartile1_M))+1
Quadrant1Quartile2_V <-as.vector(t(Quadrant1Quartile2_M))+1
Quadrant1Quartile3_V <-as.vector(t(Quadrant1Quartile3_M))+1
Quadrant1Quartile4_V <-as.vector(t(Quadrant1Quartile4_M))+1

Quadrant1Quartile4_V

MNIST_Train$Q1Q1 <- rowSums(MNIST_Train[Quadrant1Quartile1_V], na.rm=TRUE)
MNIST_Train$Q1Q2 <- rowSums(MNIST_Train[Quadrant1Quartile2_V], na.rm=TRUE)
MNIST_Train$Q1Q3 <- rowSums(MNIST_Train[Quadrant1Quartile3_V], na.rm=TRUE)
MNIST_Train$Q1Q4 <- rowSums(MNIST_Train[Quadrant1Quartile4_V], na.rm=TRUE)

#Quadrant 2


Quadrant2Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant2Quartile4_M <- matrix(,nrow=7,ncol=7)


Quadrant2Quartile1_M[1,] <- c(15:21)
Quadrant2Quartile2_M[1,] <- c(22:28)
Quadrant2Quartile3_M[1,] <- c(211:217)
Quadrant2Quartile4_M[1,] <- c(218:224)

for(i in 2:7){
  Quadrant2Quartile1_M[i,] <- Quadrant2Quartile1_M[i-1,]+28
  Quadrant2Quartile2_M[i,] <- Quadrant2Quartile2_M[i-1,]+28
  Quadrant2Quartile3_M[i,] <- Quadrant2Quartile3_M[i-1,]+28
  Quadrant2Quartile4_M[i,] <- Quadrant2Quartile4_M[i-1,]+28
}

Quadrant2Quartile1_V <-as.vector(t(Quadrant2Quartile1_M))+1
Quadrant2Quartile2_V <-as.vector(t(Quadrant2Quartile2_M))+1
Quadrant2Quartile3_V <-as.vector(t(Quadrant2Quartile3_M))+1
Quadrant2Quartile4_V <-as.vector(t(Quadrant2Quartile4_M))+1

MNIST_Train$Q2Q1 <- rowSums(MNIST_Train[Quadrant1Quartile1_V], na.rm=TRUE)
MNIST_Train$Q2Q2 <- rowSums(MNIST_Train[Quadrant1Quartile2_V], na.rm=TRUE)
MNIST_Train$Q2Q3 <- rowSums(MNIST_Train[Quadrant1Quartile3_V], na.rm=TRUE)
MNIST_Train$Q2Q4 <- rowSums(MNIST_Train[Quadrant1Quartile4_V], na.rm=TRUE)

#Quadrant 3


Quadrant3Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant3Quartile4_M <- matrix(,nrow=7,ncol=7)

Quadrant3Quartile1_M[1,] <- Q3Matrix[1,1:7]
Quadrant3Quartile2_M[1,] <- Q3Matrix[1,8:14]
Quadrant3Quartile3_M[1,] <- Q3Matrix[8,1:7]
Quadrant3Quartile4_M[1,] <- Q3Matrix[8,8:14]

for(i in 2:7){
  Quadrant3Quartile1_M[i,] <- Quadrant3Quartile1_M[i-1,]+28
  Quadrant3Quartile2_M[i,] <- Quadrant3Quartile2_M[i-1,]+28
  Quadrant3Quartile3_M[i,] <- Quadrant3Quartile3_M[i-1,]+28
  Quadrant3Quartile4_M[i,] <- Quadrant3Quartile4_M[i-1,]+28
}

Quadrant3Quartile1_V <-as.vector(t(Quadrant3Quartile1_M))+1
Quadrant3Quartile2_V <-as.vector(t(Quadrant3Quartile2_M))+1
Quadrant3Quartile3_V <-as.vector(t(Quadrant3Quartile3_M))+1
Quadrant3Quartile4_V <-as.vector(t(Quadrant3Quartile4_M))+1

MNIST_Train$Q3Q1 <- rowSums(MNIST_Train[Quadrant3Quartile1_V], na.rm=TRUE)
MNIST_Train$Q3Q2 <- rowSums(MNIST_Train[Quadrant3Quartile2_V], na.rm=TRUE)
MNIST_Train$Q3Q3 <- rowSums(MNIST_Train[Quadrant3Quartile3_V], na.rm=TRUE)
MNIST_Train$Q3Q4 <- rowSums(MNIST_Train[Quadrant3Quartile4_V], na.rm=TRUE)

#Quadrant 4

Quadrant4Quartile1_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile2_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile3_M <- matrix(,nrow=7,ncol=7)
Quadrant4Quartile4_M <- matrix(,nrow=7,ncol=7)

Quadrant4Quartile1_M[1,] <- Q4Matrix[1,1:7]
Quadrant4Quartile2_M[1,] <- Q4Matrix[1,8:14]
Quadrant4Quartile3_M[1,] <- Q4Matrix[8,1:7]
Quadrant4Quartile4_M[1,] <- Q4Matrix[8,8:14]

for(i in 2:7){
  Quadrant4Quartile1_M[i,] <- Quadrant4Quartile1_M[i-1,]+28
  Quadrant4Quartile2_M[i,] <- Quadrant4Quartile2_M[i-1,]+28
  Quadrant4Quartile3_M[i,] <- Quadrant4Quartile3_M[i-1,]+28
  Quadrant4Quartile4_M[i,] <- Quadrant4Quartile4_M[i-1,]+28
}

Quadrant4Quartile1_V <-as.vector(t(Quadrant4Quartile1_M))+1
Quadrant4Quartile2_V <-as.vector(t(Quadrant4Quartile2_M))+1
Quadrant4Quartile3_V <-as.vector(t(Quadrant4Quartile3_M))+1
Quadrant4Quartile4_V <-as.vector(t(Quadrant4Quartile4_M))+1

MNIST_Train$Q4Q1 <- rowSums(MNIST_Train[Quadrant4Quartile1_V], na.rm=TRUE)
MNIST_Train$Q4Q2 <- rowSums(MNIST_Train[Quadrant4Quartile2_V], na.rm=TRUE)
MNIST_Train$Q4Q3 <- rowSums(MNIST_Train[Quadrant4Quartile3_V], na.rm=TRUE)
MNIST_Train$Q4Q4 <- rowSums(MNIST_Train[Quadrant4Quartile4_V], na.rm=TRUE)

##Make new variables PartN -adding
#the values of the 49 parts with the x and y axis
#equally divided by 7 in the 28x28 matrix

Subset_df <- matrix(,nrow=49,ncol=16)

Subset_df[1,] <- c(1:4,29:32,57:60,85:88)
Subset_df

for (i in 2:7){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[8,] <- Subset_df[1,]+112

for (i in 9:14){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[15,] <- Subset_df[8,]+112

for (i in 16:21){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[22,] <- Subset_df[15,]+112

for (i in 23:28){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[29,] <- Subset_df[22,]+112

for (i in 30:35){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[36,] <- Subset_df[29,]+112

for (i in 37:42){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df[43,] <- Subset_df[36,] +112

for (i in 44:49){
  Subset_df[i,] <- Subset_df[i-1,]+4
}

Subset_df <- Subset_df+1

MNIST_Train$Subset1 <- rowSums(MNIST_Train[Subset_df[1,]], na.rm=TRUE)
MNIST_Train$Subset2 <- rowSums(MNIST_Train[Subset_df[2,]], na.rm=TRUE)
MNIST_Train$Subset3 <- rowSums(MNIST_Train[Subset_df[3,]], na.rm=TRUE)
MNIST_Train$Subset4 <- rowSums(MNIST_Train[Subset_df[4,]], na.rm=TRUE)
MNIST_Train$Subset5 <- rowSums(MNIST_Train[Subset_df[5,]], na.rm=TRUE)
MNIST_Train$Subset6 <- rowSums(MNIST_Train[Subset_df[6,]], na.rm=TRUE)
MNIST_Train$Subset7 <- rowSums(MNIST_Train[Subset_df[7,]], na.rm=TRUE)
MNIST_Train$Subset8 <- rowSums(MNIST_Train[Subset_df[8,]], na.rm=TRUE)
MNIST_Train$Subset9 <- rowSums(MNIST_Train[Subset_df[9,]], na.rm=TRUE)

MNIST_Train$Subset10 <- rowSums(MNIST_Train[Subset_df[10,]], na.rm=TRUE)
MNIST_Train$Subset11 <- rowSums(MNIST_Train[Subset_df[11,]], na.rm=TRUE)
MNIST_Train$Subset12 <- rowSums(MNIST_Train[Subset_df[12,]], na.rm=TRUE)
MNIST_Train$Subset13 <- rowSums(MNIST_Train[Subset_df[13,]], na.rm=TRUE)
MNIST_Train$Subset14 <- rowSums(MNIST_Train[Subset_df[14,]], na.rm=TRUE)
MNIST_Train$Subset15 <- rowSums(MNIST_Train[Subset_df[15,]], na.rm=TRUE)
MNIST_Train$Subset16 <- rowSums(MNIST_Train[Subset_df[16,]], na.rm=TRUE)
MNIST_Train$Subset17 <- rowSums(MNIST_Train[Subset_df[17,]], na.rm=TRUE)
MNIST_Train$Subset18 <- rowSums(MNIST_Train[Subset_df[18,]], na.rm=TRUE)
MNIST_Train$Subset19 <- rowSums(MNIST_Train[Subset_df[19,]], na.rm=TRUE)
MNIST_Train$Subset20 <- rowSums(MNIST_Train[Subset_df[20,]], na.rm=TRUE)
MNIST_Train$Subset21 <- rowSums(MNIST_Train[Subset_df[21,]], na.rm=TRUE)

MNIST_Train$Subset22 <- rowSums(MNIST_Train[Subset_df[22,]], na.rm=TRUE)
MNIST_Train$Subset23 <- rowSums(MNIST_Train[Subset_df[23,]], na.rm=TRUE)
MNIST_Train$Subset24 <- rowSums(MNIST_Train[Subset_df[24,]], na.rm=TRUE)
MNIST_Train$Subset25 <- rowSums(MNIST_Train[Subset_df[25,]], na.rm=TRUE)
MNIST_Train$Subset26 <- rowSums(MNIST_Train[Subset_df[26,]], na.rm=TRUE)
MNIST_Train$Subset27 <- rowSums(MNIST_Train[Subset_df[27,]], na.rm=TRUE)
MNIST_Train$Subset28 <- rowSums(MNIST_Train[Subset_df[28,]], na.rm=TRUE)
MNIST_Train$Subset29 <- rowSums(MNIST_Train[Subset_df[29,]], na.rm=TRUE)
MNIST_Train$Subset30 <- rowSums(MNIST_Train[Subset_df[30,]], na.rm=TRUE)
MNIST_Train$Subset31 <- rowSums(MNIST_Train[Subset_df[31,]], na.rm=TRUE)
MNIST_Train$Subset32 <- rowSums(MNIST_Train[Subset_df[32,]], na.rm=TRUE)
MNIST_Train$Subset33 <- rowSums(MNIST_Train[Subset_df[33,]], na.rm=TRUE)
MNIST_Train$Subset34 <- rowSums(MNIST_Train[Subset_df[34,]], na.rm=TRUE)
MNIST_Train$Subset35 <- rowSums(MNIST_Train[Subset_df[35,]], na.rm=TRUE)
MNIST_Train$Subset36 <- rowSums(MNIST_Train[Subset_df[36,]], na.rm=TRUE)
MNIST_Train$Subset37 <- rowSums(MNIST_Train[Subset_df[37,]], na.rm=TRUE)
MNIST_Train$Subset38 <- rowSums(MNIST_Train[Subset_df[38,]], na.rm=TRUE)
MNIST_Train$Subset39 <- rowSums(MNIST_Train[Subset_df[39,]], na.rm=TRUE)
MNIST_Train$Subset40 <- rowSums(MNIST_Train[Subset_df[40,]], na.rm=TRUE)
MNIST_Train$Subset41 <- rowSums(MNIST_Train[Subset_df[41,]], na.rm=TRUE)
MNIST_Train$Subset42 <- rowSums(MNIST_Train[Subset_df[42,]], na.rm=TRUE)

MNIST_Train$Subset43 <- rowSums(MNIST_Train[Subset_df[43,]], na.rm=TRUE)
MNIST_Train$Subset44 <- rowSums(MNIST_Train[Subset_df[44,]], na.rm=TRUE)
MNIST_Train$Subset45 <- rowSums(MNIST_Train[Subset_df[45,]], na.rm=TRUE)
MNIST_Train$Subset46 <- rowSums(MNIST_Train[Subset_df[46,]], na.rm=TRUE)
MNIST_Train$Subset47 <- rowSums(MNIST_Train[Subset_df[47,]], na.rm=TRUE)
MNIST_Train$Subset48 <- rowSums(MNIST_Train[Subset_df[48,]], na.rm=TRUE)
MNIST_Train$Subset49 <- rowSums(MNIST_Train[Subset_df[49,]], na.rm=TRUE)


#Creates a dataset that has the number of pixels that is above 0


MNIST_Train$Pixel_Number <- rowSums(MNIST_Train > 0)
MNIST_Test$Pixel_Number <- rowSums(MNIST_Test > 0)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Exports the new dataset into a CSV
write.csv(MNIST_Test, file = "MNIST_Test2.csv")
write.csv(MNIST_Train, file = "MNIST_Train2.csv")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Cleaning the workspace - remove functions and data

rm(image1_m,image2_m,image3_m,image4_m)
rm(LeftHalf_Matrix,RightHalf_Matrix,Q1Matrix,Q2Matrix,Q3Matrix,Q4Matrix)
rm(LeftHalf_Vector,RightHalf_Vector,Q1_Vector,Q2_Vector,Q3_Vector,Q4_Vector)

rm(MNIST_Colnames,MNIST_Colnames_Lowerhalf,MNIST_Colnames_Upperhalf,i)

rm(Quadrant1Quartile1_M,Quadrant1Quartile2_M,Quadrant1Quartile3_M,Quadrant1Quartile4_M)
rm(Quadrant2Quartile1_M,Quadrant2Quartile2_M,Quadrant2Quartile3_M,Quadrant2Quartile4_M)
rm(Quadrant3Quartile1_M,Quadrant3Quartile2_M,Quadrant3Quartile3_M,Quadrant3Quartile4_M)
rm(Quadrant4Quartile1_M,Quadrant4Quartile2_M,Quadrant4Quartile3_M,Quadrant4Quartile4_M)

rm(Quadrant1Quartile1_V,Quadrant1Quartile2_V,Quadrant1Quartile3_V,Quadrant1Quartile4_V)
rm(Quadrant2Quartile1_V,Quadrant2Quartile2_V,Quadrant2Quartile3_V,Quadrant2Quartile4_V)
rm(Quadrant3Quartile1_V,Quadrant3Quartile2_V,Quadrant3Quartile3_V,Quadrant3Quartile4_V)
rm(Quadrant4Quartile1_V,Quadrant4Quartile2_V,Quadrant4Quartile3_V,Quadrant4Quartile4_V)

rm(rotate,QuartileAddition,QuartileAddition,Subset_df)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


#Create the training set that uses only new data
MNIST_Train_New <-MNIST_Train[,c(1,786:860)]
MNIST_Test_New <- MNIST_Test[,c(1,786:860)]


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


#Creating the model

#1st Objective: Build a classifier using all pixels as features for handwriting recognition.

#Making a Multiple Linear Regression Model using all the pixels

#Make a subset out of the x variables in the training set
MNIST_Train_X <- MNIST_Train[2:785]

#Define the algorithm

MNIST_Train_Y <-MNIST_Train$Label

fit_linear_train <- lm(MNIST_Train_Y~., MNIST_Train_X )

#Create the model
fit_linear_train

#Make a subset ouf the x variables in the test set
MNIST_Test_X <- MNIST_Test[2:785]

#Use the model to predict the outcome of the test set
fit_linear_test <- predict(fit_linear_train,MNIST_Test_X)

#run fit_linear_test to get the predicted y -values and round them to the nearest integer
#Create a dataset of the true y-values
fit_linear_test
rounded_fl_test <- as.data.frame(round(fit_linear_test))
rounded_fl_test[1]
MNIST_Test_Y <- MNIST_Test[1]
MNIST_Test[1]

#Compare the predicted y values to the real y values
Result_fl_test <- rounded_fl_test == MNIST_Test_Y
summary(Result_fl_test)


#Do Multiple Linear Regression with the new data only

fit_linear_train2 <- lm(MNIST_Train_New[,1]~., MNIST_Train_New[,-1])
fit_linear_test2 <- predict(fit_linear_train2,MNIST_Test_New[,-1])
fit_linear_test2.1 <- as.data.frame(round(fit_linear_test2))
Result_fl_test2 <- fit_linear_test2.1 == MNIST_Test$Label
summary(Result_fl_test2)



#Do Multiple Linear Regression wiht new data and original data

fit_linear_train3 <- lm(MNIST_Train[,1]~., MNIST_Train[,-1])
fit_linear_test3 <- predict(fit_linear_train3,MNIST_Test[,-1])
fit_linear_test3.1 <- as.data.frame(round(fit_linear_test3))
Result_fl_test3 <- fit_linear_test3.1 == MNIST_Test$Label
summary(Result_fl_test3)


#Objective 2: B. Compare a number of different classifiers (such as logistic regression and SVM) and 
#compare their pros and cons for handwriting accuracy

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
ridgeseq=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Ridge regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
ridgecv.out=cv.glmnet(x,y,alpha=0,lambda=ridgeseq,family="multinomial",nfolds=10)
ridgebestlam=ridgecv.out$lambda.min

#Build Ridge model using original training dataset
ridge=glmnet(x,y,alpha=0, lambda=ridgebestlam, family="multinomial")

#Evaluate this model on the test set
ridgepredt=predict(ridge, xt,type="class")
table(ridgepredt,yt)

#Error rate
sum(ridgepredt!= yt)/length(ridgepredt)
mean(ridgepredt!=yt)

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
lassoseq=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Lasso regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
lassocv.out=cv.glmnet(x,y,alpha=1,lambda=lassoseq,family="multinomial",nfolds=10)
lassobestlam=lassocv.out$lambda.min

#Build Lasso model using original training dataset
lasso=glmnet(x,y,alpha=1, lambda=lassobestlam, family="multinomial")

#Evaluate this model on the testing dataset
lassopredt=predict(lasso, xt,type="class")
table(lassopredt,yt)

#Error rate
sum(lassopredt!= yt)/length(lassopredt)
mean(lassopredt!=yt)

#Prepare the arguments for glmnet() on the new features of processed dataset
x2=MINSTTRAIN2[,786:859]
x2=as.matrix(x2)
y2=as.factor(MINSTTRAIN2$Label)
xt2=MINSTTEST2[,786:859]
xt2=as.matrix(xt2)
yt2=as.factor(MINSTTEST2$Label)

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
ridgeseq2=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Ridge regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
ridgecv.out2=cv.glmnet(x2,y2,alpha=0,lambda=ridgeseq2,family="multinomial",nfolds=10)
ridgebestlam2=ridgecv.out2$lambda.min

#Build Ridge model using processed training dataset
ridge2=glmnet(x2,y2,alpha=0, lambda=ridgebestlam2, family="multinomial")

#Evaluate this model on the processed testing dataset
ridgepredt2=predict(ridge2, xt2,type="class")
table(ridgepredt2,yt2)

#Error rate
sum(ridgepredt2!= yt2)/length(ridgepredt2)
mean(ridgepredt2!=yt2)

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
lassoseq2=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Lasso regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
lassocv.out2=cv.glmnet(x2,y2,alpha=1,lambda=lassoseq2,family="multinomial",nfolds=10)
lassobestlam2=lassocv.out2$lambda.min

#Build Lasso model using processed training dataset
lasso2=glmnet(x2,y2,alpha=1, lambda=lassobestlam2, family="multinomial")

#Evaluate this model on the processed testing dataset
lassopredt2=predict(lasso2, xt2,type="class")
table(lassopredt2,yt2)

#Error rate
sum(lassopredt2!= yt2)/length(lassopredt2)
mean(lassopredt2!=yt2)

#Prepare the arguments for glmnet() on all features of processed dataset
x3=model.matrix(MINSTTRAIN2$Label~.,MINSTTRAIN2)
y3=as.factor(MINSTTRAIN2$Label)
xt3=model.matrix(MINSTTEST2$Label~.,MINSTTEST2)
yt3=as.factor(MINSTTEST2$Label)

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
ridgeseq3=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Ridge regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
ridgecv.out3=cv.glmnet(x3,y3,alpha=0,lambda=ridgeseq3,family="multinomial",nfolds=10)
ridgebestlam3=ridgecv.out3$lambda.min

#Build Ridge model using processed training dataset
ridge3=glmnet(x3,y3,alpha=0, lambda=ridgebestlam3, family="multinomial")

#Evaluate this model on the processed testing dataset
ridgepredt3=predict(ridge3, xt3,type="class")
table(ridgepredt3,yt3)

#Error rate
sum(ridgepredt3!= yt3)/length(ridgepredt3)
mean(ridgepredt3!=yt3)

#Ridge and Lasso regression have a tuneable parameter: lambda
#As the dataset is large, we would choose the best model using CV among lambda=10^-1,10^3,...,10^10
lassoseq3=10^(-1:10) #set sequence of lambdas we want to test

#Use 10-fold CV to choose the best value of lambda for Lasso regression
#For the command below, alpha=0: ridge regression, alpha=1: lasso regression
#Famliy is multinomial because we need to perform logistic regression that the response is multinomial
lassocv.out3=cv.glmnet(x3,y3,alpha=1,lambda=lassoseq3,family="multinomial",nfolds=10)
lassobestlam3=lassocv.out3$lambda.min

#Build Lasso model using processed training dataset
lasso3=glmnet(x3,y3,alpha=1, lambda=lassobestlam3, family="multinomial")

#Evaluate this model on processed testing dataset
lassopredt3 = predict(lasso3, xt3,type="class")
table(lassopredt3,yt3)

#Error rate
sum(lassopredt3!= yt3)/length(lassopredt3)
mean(lassopredt3!=yt3)


##########SVM Model
#install SVM package in R
install.packages("e1071")

#Read the package
library(e1071)

#Prepare the arguments for svm on new features, we select 10000 rows of data from the training dataset to tune svm model because it will take super long time to tune on 60000 rows of data
x4<-MINSTTRAIN2[2:10001,786:859]
x4<-as.matrix(x4)
y4<-MINSTTRAIN2[2:10001,1]
xt4<-MINSTTEST2[,786:859]
xt4<-as.matrix(xt4)
yt4<-MINSTTEST2[,1]

#Tune the selected data to find out the best parametres and smooth the model
svm_tune<-tune(svm,x4,y4,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#We find that the best parametres would be cost=10 and gamma=0.5

#Randomly select samples from the training dataset to build svm model because it will take super long time to build svm model on 60000 rows of data
TRAIN2sample3<-MINSTTRAIN2[sample(nrow(MINSTTRAIN2),10000),c(1,786:859)]
TEST2sample3<-MINSTTEST2[,c(2,786:859)]

#Build SVM model, setting "label" as dependent variable and pixels as independent variables, and record the time 
svmmodel_tune<-svm(TRAIN2sample3$Label~., data=TRAIN2sample3, kernel="radial", cost=10, gamma=0.5,scale=F)
summary(svmmodel_tune)
predsvm2<-predict(svmmodel_tune,xt4)
system.time(predict(svmmodel_tune,xt4))
predsvm2rounded<-round(predsvm2)
predsvm2result<-predsvm2rounded == yt4
summary(predsvm2result)
table(predsvm2rounded,yt4)

#Prepare the arguments for svm on all features, we select 10000 rows of data from the training dataset to tune svm model because it will take super long time to tune on 60000 rows of data
x5<-MINSTTRAIN2[10002:20001,2:859]
x5<-as.matrix(x5)
y5<-MINSTTRAIN2[10002:20001,1]
xt5<-MINSTTEST2[,2:859]
xt5<-as.matrix(xt5)
yt5<-MINSTTEST2[,1]

#Tune the data to select the best parametres and smooth the model
svm_tune2<-tune(svm,x5,y5,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune2)
#We find that the best parametres would be cost=10 and gamma=0.5

#Randomly select samples from the training dataset to build svm model because it will take super long time to build svm model on 60000 rows of data
TRAIN2sample4<-MINSTTRAIN2[sample(nrow(MINSTTRAIN2),10000),c(1,786:859)]
TEST2sample4<-MINSTTEST2[,c(1,786:859)]

#Build SVM model, setting "label" as dependent variable and pixels as independent variables, and record the time 
svmmodel_tune2<-svm(TRAIN2sample4$Label~., data=TRAIN2sample4, kernel="radial", cost=10, gamma=0.5,scale=F)
summary(svmmodel_tune2)
predsvm3<-predict(svmmodel_tune2,xt5)
system.time(predict(svmmodel_tune2,xt5))
predsvm3rounded<-round(predsvm3)
predsvm3result<-predsvm3rounded == yt5
summary(predsvm3result)
table(predsvm3rounded,yt5)

#Prepare the arguments for svm on all features on all data
x6<-MINSTTRAIN2[,2:859]
x6<-as.matrix(x6)
y6<-MINSTTRAIN2[,1]
xt6<-MINSTTEST2[,2:859]
xt6<-as.matrix(xt6)
yt6<-MINSTTEST2[,1]

#Tune the selected data to find out the best parametres and smooth the model
svm_tune3<-tune(svm,x6,y6,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune3)
#We find that the best parametres would be cost=10 and gamma=0.5

#Build SVM model, setting "label" as dependent variable and pixels as independent variables, and record the time 
svmmodel_tune3<-svm(TRAIN2$Label~., data=TRAIN2, kernel="radial", cost=10, gamma=0.5,scale=F)
summary(svmmodel_tune3)
predsvm4<-predict(svmmodel_tune3,xt6)
system.time(predict(svmmodel_tune3,xt6))
predsvm4rounded<-round(predsvm4)
predsvm4result<-predsvm4rounded == yt6
summary(predsvm4result)
table(predsvm2rounded,yt6)


#Decision Tree

MNIST_Train_Original <- MNIST_Train[1:785]
MNIST_Test_Original <- MNIST_Test[1:785]
library(rpart)
fit_decision_train <- rpart(Label~.,data=MNIST_Train_Original,method="class")
fit_decision_train

library(rattle)
fancyRpartPlot(fit_decision_train)
library(rpart.plot)

plot (fit_decision_train)
text(fit_decision_train, use.n=TRUE)

fit_decision_test <- predict(fit_decision_train,MNIST_Test_Original,type = "class")
fit_decision_test <- as.data.frame(fit_decision_test)
Result_decision_test <- fit_decision_test == as.numeric(MNIST_Test$Label)

summary (Result_decision_test)

#Create the training set that uses only new data
MNIST_Train_New <-MNIST_Train[,c(1,786:859)]
MNIST_Test_New <- MNIST_Test[,c(1,786:859)]


fit_decision_train2 <- rpart(Label~.,data=MNIST_Train_New,method="class")
fit_decision_test2 <- predict(fit_decision_train2,MNIST_Test_New,type = "class")
fit_decision_test2 <- as.data.frame(fit_decision_test2)
Result_decision_test2 <- fit_decision_test2 == as.numeric(MNIST_Test$Label)
summary (Result_decision_test2)

#Create the training set that uses complete data

fit_decision_train3 <- rpart(Label~.,data=MNIST_Train,method="class")
fit_decision_test3 <- predict(fit_decision_train3,MNIST_Test,type = "class")
fit_decision_test3 <- as.data.frame(fit_decision_test3)
Result_decision_test3 <- fit_decision_test3 == as.numeric(MNIST_Test$Label)
summary (Result_decision_test3)



#Random Forest
MNIST.New.Only <- MNIST_Train[,c(1,796:860)]

library(randomForest)
MNIST_Train_Original$Label <- as.factor(MNIST_Train_Original$Label)

fit_rf_train <- randomForest(Label~.,data=MNIST_Train_Original, ntree=100)

fit_rf_test <- predict(fit_rf_train,MNIST_Test)
fit_rf_test <- as.data.frame(fit_rf_test)
Result_decision_test_rf <- fit_rf_test == as.numeric(MNIST_Test$Label)

summary (Result_decision_test_rf)

#RF with only new data

fit_rf_train2 <- randomForest(Label~.,data=MNIST_Train_New, ntree=100)
fit_rf_test2 <- predict(fit_rf_train2,MNIST_Test_New)
fit_rf_test2 <- as.data.frame(fit_rf_test2)
Result_decision_test_rf2 <- fit_rf_test2 == as.numeric(MNIST_Test$Label)

summary(Result_decision_test_rf2)


#RF with complete data

fit_rf_train3 <- randomForest(Label~.,data=MNIST_Train, ntree=100)
fit_rf_test3 <- predict(fit_rf_train3,MNIST_Test)
fit_rf_test3 <- as.data.frame(fit_rf_test3)
Result_decision_test_rf23 <- fit_rf_test3 == as.numeric(MNIST_Test$Label)

summary(Result_decision_test_rf3)


#K-Means

fit_kmeans_test <- kmeans(c(MNIST_Test$Right_Half,MNIST_Test$Left_Half),10, iter.max=100,nstart=10)

table(data.frame(fit_kmeans_test$cluster, MNIST_Test$Label))

#Calculate Confusion

calculate.confusion <- function(states, clusters)
{
  # generate a confusion matrix of cols C versus states S
  d <- data.frame(state = states, cluster = clusters)
  td <- as.data.frame(table(d))
  # convert from raw counts to percentage of each label
  pc <- matrix(ncol=max(clusters),nrow=0) # k cols
  for (i in 1:10) # 10 labels
  {
    total <- sum(td[td$state==td$state[i],3])
    pc <- rbind(pc, td[td$state==td$state[i],3]/total)
  }
  rownames(pc) <- td[1:10,1]
  return(pc)
}

#Assign Cluster Labels

assign.cluster.labels <- function(cm, k)
{
  # take the cluster label from the highest percentage in that column
  cluster.labels <- list()
  for (i in 1:k)
  {
    cluster.labels <- rbind(cluster.labels, row.names(cm)[match(max(cm[,i]), cm[,i])])
  }
  
  # this may still miss some labels, so make sure all labels are included
  for (l in rownames(cm)) 
  { 
    if (!(l %in% cluster.labels)) 
    { 
      cluster.number <- match(max(cm[l,]), cm[l,])
      cluster.labels[[cluster.number]] <- c(cluster.labels[[cluster.number]], l)
    } 
  }
  return(cluster.labels)
}

#Creates the list of cluster labels

str(assign.cluster.labels(calculate.confusion(MNIST_Test$Label, fit_kmeans_test$cluster), 10))

#Calculate Accuracy

calculate.accuracy <- function(states, clabels)
{
  matching <- Map(function(state, labels) { state %in% labels }, states, clabels)
  tf <- unlist(matching, use.names=FALSE)
  return (sum(tf)/length(tf))
}

#Run the functions to get the accuracy

k <- length(fit_kmeans_test$size)
conf.mat <- calculate.confusion(MNIST_Test$Label, fit_kmeans_test$cluster)
cluster.labels <- assign.cluster.labels(conf.mat, k)
acc <- calculate.accuracy(MNIST_Test$Label, cluster.labels[fit_kmeans_test$cluster])
cat("For", k, "means with accuracy", acc, ", labels are assigned as:\n")
cat(str(cluster.labels))


#Saving the environment

save.image(file="ML Midterm Project.RData")
load(file="ML Midterm Project.RData")
