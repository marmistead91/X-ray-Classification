# X-ray-Classification

## Background

I created an CNN image classifaction model that looks at chest x rays and determines if the lungs are healthy or have pneumonia. The reason I chose this project is becasue pneumonia is one of the complications that leads to death with Covid 19. I am not doing this to replace doctors but to help aid them. 

What is Pneumonia? Pneumonia is lung inflammation caused by bacterial or viral infection, in which the air sacs fill with pus and may become solid. 
When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection.

Healthy lungs
![](X-ray%20Eda%20pics/normal4x4.png)

If you look closely you can see some of the white spots in the lungs
Lungs with Pneumonia
![](X-ray%20Eda%20pics/pneu4x4.png)

## EDA

I have 5,863 X-Ray images from Kaggle(https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The images are anterior-posterior chest X rays of pediatric patients that are between 1 and 5 years old. When I did my research pneumonia is going to look the same in both adults and children. The difference in the X rays is going to be the development of the organs. The diagnoses of the images were graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert. 

My data was heavily imbalanced as you can see and I have more pneumonia images than healthy. The baseline accuracy score for my testing data is 62.5%

![](X-ray%20Eda%20pics/test_distribution.png) ![](X-ray%20Eda%20pics/train_distribution.png)

