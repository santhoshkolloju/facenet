Face Recognition in Tensorflow using prebuilt model from facenet

Finding the threshold for similarity

Create a folder Test_Images in the master folder
Arrange your images in the following way
<pre>
Step1:
Test_Images
      Mahesh Babu
          image1.jpg
          image2.jpg
          image3.jpg
      Pawan Kalyan
          image1.jpg
          image2.jpg
          image3.jpg
      Allu Arjun
          image1.jpg
          image2.jpg
          image3.jpg
Names need not be the way mentioned can have any name


Step2:
Create a folder named Face_distance_matrix in the master folder

Step3:
python src/compare.py Test_Images/*

Step4:
python src/find_threshold.py 80 80
this gives two ouptus
1)what should be the threshold cutoff if we want to identify similar images with 80% accuracy
2)What should be cutoff threshold if we want to identify no similar images with 80% accuracy 

Change percentage according to your needs.

</pre>
