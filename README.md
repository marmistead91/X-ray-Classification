# X-ray-Classification

## Background

I created an CNN image classification model that looks at chest x rays and determines if the lungs are healthy or have pneumonia. The reason I chose this project is because pneumonia is one of the complications that leads to death with Covid 19. I am not doing this to replace doctors but to help aid them. 

Pneumonia is lung inflammation caused by bacterial or viral infection, in which the air sacs fill with pus and may become solid. 
When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection.

Healthy lungs
![](X-ray%20Eda%20pics/normal4x4.png)

If you look closely you can see some of the white spots in the lungs

Lungs with Pneumonia
![](X-ray%20Eda%20pics/pneu4x4.png)

## EDA

I have 5,863 X-Ray images from Kaggle (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The images are anterior-posterior chest X rays of pediatric patients that are between 1 and 5 years old. When I did my research pneumonia is going to look the same in both adults and children. The difference in the X rays is going to be the development of the organs. The diagnoses of the images were graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert. 

My data was heavily imbalanced as you can see and I have more pneumonia images than healthy. The baseline accuracy score for my testing data is 62.5%

![](X-ray%20Eda%20pics/test_distribution.png) ![](X-ray%20Eda%20pics/train_distribution.png)

The images came in a wide range of resolutions with the mean being 1327x970. The doctors are going to want this images in high resolution so they can easily find any pneumonia.

![](X-ray%20Eda%20pics/imgsize_scatter.png)

Trained my models at different sizes of the images, 256x256 and 128x128. The 128x128 size ended up with a better outcome.

## Models

Binary CNN model that predicted if the lungs were healthy or if they had pneumonia. My first model was a basic CNN model with 4 layers of convolution and max-pooling using Relu then densing and a dropout rate of .1
Next I used the same model but this time I augmented the training data to see if there would be a better outcome. I used ImageDataGenorator to change width_shift, height_shift, zoom, shear_range and flip.
The last model used transfer learning with VGG19 and data augmentation. I wasn't sure how this was going to work because VGG19 is trained on millions of natural images and not medical images.

## Evaluation

Using 4 different evaluation metrics, Accuracy, Precision, Recall, F1 score. Looking at the confusion matrixes the top left and bottom right at the correct predictions. The top right(False Positives) and bottom left(False Negatives) are incorrect, top right(FP) predicted that they were had pneumonia but were actually healthy and the bottom left(FN) predicted that they were healthy when they actually had pneumonia. I would rather have more incorrectly guessing they are sick when they are healthy than the other way around.

Basic model
  Accuracy : 78.37%
  Recall : 0.43
  Precision : 0.99
  F1 : 0.6
  
  ![](X-ray-Classification/blob/master/X-ray%20Eda%20pics/CM_bassic.png)
  
Basic model with Data Augmentation to images
  Accuracy : 87.18%
  Recall : 0.91
  Precision : 0.78
  F1 : 0.84
  
  ![](X-ray-Classification/blob/master/X-ray%20Eda%20pics/CM_aug.png)
  
Transfer Learning with Data Augmentation to images
  Accuracy : 90.71%
  Recall : 0.78
  Precision : 0.97
  F1 : 0.86
  
  ![](X-ray-Classification/blob/master/X-ray%20Eda%20pics/CM_tl.png)
  
## Insight and Analysis

Based off of accuracy, F1 score and the low False Negatives the transfer learning model with data augmentation would be the best model. The next best would be the basic model because of the low False Negatives and high Precision. 

## Further Exploration

This model needs more images and images of older patients before it can be used in the real world. I would also like to use a stronger computer and see if keeping the images at a higher resolution could help. 
