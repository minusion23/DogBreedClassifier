Problem Statement
The following repository has been created to track the work made in relation to a neural netrowk classification problems and the use of a solution in a web app. Both problems were challenging,
one requiring training an effective model for a given classificaiton problem and the other requiring running a neural network on a server and effectively transferring information
between the index and a rest-api set up to answer the queries.


Model training and preparation

The first part of the project (dog_app.ipynb, dog_app.htnl) was comprised of working with data sets containing labeled images of dogs(labels were including the dog's breed), as well as pictures of individuals.
An effort was made there to work on pre-processing the images and prepare a neural netowrk built from scratch that could effectively predict the breed of the dog in an image.
Data including human faces was used as a baseline and later on as an additional feature of the application. The face identification was run using CV2 built in library/model.

After much thought put into setting up a custom neural network, additional effort was made to use transfer learning to achieve better results/accuracy calculated with
the percent of correctly identified images from a test set that the neural network previously has not interacted with.
ResNet50 has been chosen as the model to use with the transfer learning approach. This model had it's top sliced off so that the model's bottleneck features can be used as 
input for a custom model used to identify the breeds (this greatly reduce the time needed to arrive at good results).

Training Results

A satisfying accuracy of over 80% on the test set has been achieved. Later an algorithm has been set up so that based on the provided image the model would provied
the predicted dog bread, a breed most similar to an individual on the picture (if a face was identified in the image), or an error if neither of the two was identified.

A web implementation of the model

The algorithm and the trained model served as base to create a web app using BootStrap, JQuery, Flask. The application allows the user to upload an image to the website
which was shown to the user together with a prediction that the models working on the back returned. Quite interestingly, there were a few challenges here, including choosing correct
dependencies since the model was originally trained on a ResNet50 version of which output was (1,1), while the newer version is (7,7). 
Quite interestingly, maybe due to a slightly different handling and pre-processing of the images, the results of the web version did not always align in 100 percent with
the jupyter notebook version of the algorithm. 

### Instructions for the web App:
1. Run the following commands in the project's root directory to set up the model, servers and run the web app.

"python application.py"

2. Go to http://127.0.0.1:3001/

Application.py - starts the servers and set ups the models
Routes.py - stores the majority of the code including model loading, image and inference processing functions. 

Conclusion 

The algorithm is not working too bad in relation to accuracy but could be improved upon:

1. Processing the algorightm takes a long time. Maybe try to work on the uploading and transforming the image for a quicker response.
2. Increase the accuracy - work on providing more dropout layers and more challenging data so that the model does not fit so good on the training data and lifts the validation score and consequently, the final prediction
3. Prepare the model for more atypical pictures, so work on randomizing data in relation to invariance
 
