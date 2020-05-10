# Import the necessary libraries
from DogClassifierApp import app
from flask import render_template
from keras.applications import ResNet50
from PIL import Image
from PIL import ImageFile
import numpy as np
import flask
import io
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image


def load_models():
    # Set up global variables that will be used throughout the "routes". This will be done on server start up
    global model
    global model_custom
    global face_cascade
    global dog_names
    global graph

    # Load the models as well as the face classifier
    face_cascade = cv2.CascadeClassifier("dogclassifierapp/static/haarcascade_frontalface_alt.xml")
    model = ResNet50(weights= 'imagenet', include_top=False)
    model_custom = load_model("dogclassifierapp/static/ResNetDog.h5")
    # Prepare the dog_names to be retrieved after predicitions are made to provide the predicted name
    dog_names = ['ages/train/001.Affenpinscher', 'ages/train/002.Afghan_hound', 'ages/train/003.Airedale_terrier', 'ages/train/004.Akita', 'ages/train/005.Alaskan_malamute', 'ages/train/006.American_eskimo_dog', 'ages/train/007.American_foxhound', 'ages/train/008.American_staffordshire_terrier', 'ages/train/009.American_water_spaniel', 'ages/train/010.Anatolian_shepherd_dog', 'ages/train/011.Australian_cattle_dog', 'ages/train/012.Australian_shepherd', 'ages/train/013.Australian_terrier', 'ages/train/014.Basenji', 'ages/train/015.Basset_hound', 'ages/train/016.Beagle', 'ages/train/017.Bearded_collie', 'ages/train/018.Beauceron', 'ages/train/019.Bedlington_terrier', 'ages/train/020.Belgian_malinois', 'ages/train/021.Belgian_sheepdog', 'ages/train/022.Belgian_tervuren', 'ages/train/023.Bernese_mountain_dog', 'ages/train/024.Bichon_frise', 'ages/train/025.Black_and_tan_coonhound', 'ages/train/026.Black_russian_terrier', 'ages/train/027.Bloodhound', 'ages/train/028.Bluetick_coonhound', 'ages/train/029.Border_collie', 'ages/train/030.Border_terrier', 'ages/train/031.Borzoi', 'ages/train/032.Boston_terrier', 'ages/train/033.Bouvier_des_flandres', 'ages/train/034.Boxer', 'ages/train/035.Boykin_spaniel', 'ages/train/036.Briard', 'ages/train/037.Brittany', 'ages/train/038.Brussels_griffon', 'ages/train/039.Bull_terrier', 'ages/train/040.Bulldog', 'ages/train/041.Bullmastiff', 'ages/train/042.Cairn_terrier', 'ages/train/043.Canaan_dog', 'ages/train/044.Cane_corso', 'ages/train/045.Cardigan_welsh_corgi', 'ages/train/046.Cavalier_king_charles_spaniel', 'ages/train/047.Chesapeake_bay_retriever', 'ages/train/048.Chihuahua', 'ages/train/049.Chinese_crested', 'ages/train/050.Chinese_shar-pei', 'ages/train/051.Chow_chow', 'ages/train/052.Clumber_spaniel', 'ages/train/053.Cocker_spaniel', 'ages/train/054.Collie', 'ages/train/055.Curly-coated_retriever', 'ages/train/056.Dachshund', 'ages/train/057.Dalmatian', 'ages/train/058.Dandie_dinmont_terrier', 'ages/train/059.Doberman_pinscher', 'ages/train/060.Dogue_de_bordeaux', 'ages/train/061.English_cocker_spaniel', 'ages/train/062.English_setter', 'ages/train/063.English_springer_spaniel', 'ages/train/064.English_toy_spaniel', 'ages/train/065.Entlebucher_mountain_dog', 'ages/train/066.Field_spaniel', 'ages/train/067.Finnish_spitz', 'ages/train/068.Flat-coated_retriever', 'ages/train/069.French_bulldog', 'ages/train/070.German_pinscher', 'ages/train/071.German_shepherd_dog', 'ages/train/072.German_shorthaired_pointer', 'ages/train/073.German_wirehaired_pointer', 'ages/train/074.Giant_schnauzer', 'ages/train/075.Glen_of_imaal_terrier', 'ages/train/076.Golden_retriever', 'ages/train/077.Gordon_setter', 'ages/train/078.Great_dane', 'ages/train/079.Great_pyrenees', 'ages/train/080.Greater_swiss_mountain_dog', 'ages/train/081.Greyhound', 'ages/train/082.Havanese', 'ages/train/083.Ibizan_hound', 'ages/train/084.Icelandic_sheepdog', 'ages/train/085.Irish_red_and_white_setter', 'ages/train/086.Irish_setter', 'ages/train/087.Irish_terrier', 'ages/train/088.Irish_water_spaniel', 'ages/train/089.Irish_wolfhound', 'ages/train/090.Italian_greyhound', 'ages/train/091.Japanese_chin', 'ages/train/092.Keeshond', 'ages/train/093.Kerry_blue_terrier', 'ages/train/094.Komondor', 'ages/train/095.Kuvasz', 'ages/train/096.Labrador_retriever', 'ages/train/097.Lakeland_terrier', 'ages/train/098.Leonberger', 'ages/train/099.Lhasa_apso', 'ages/train/100.Lowchen', 'ages/train/101.Maltese', 'ages/train/102.Manchester_terrier', 'ages/train/103.Mastiff', 'ages/train/104.Miniature_schnauzer', 'ages/train/105.Neapolitan_mastiff', 'ages/train/106.Newfoundland', 'ages/train/107.Norfolk_terrier', 'ages/train/108.Norwegian_buhund', 'ages/train/109.Norwegian_elkhound', 'ages/train/110.Norwegian_lundehund', 'ages/train/111.Norwich_terrier', 'ages/train/112.Nova_scotia_duck_tolling_retriever', 'ages/train/113.Old_english_sheepdog', 'ages/train/114.Otterhound', 'ages/train/115.Papillon', 'ages/train/116.Parson_russell_terrier', 'ages/train/117.Pekingese', 'ages/train/118.Pembroke_welsh_corgi', 'ages/train/119.Petit_basset_griffon_vendeen', 'ages/train/120.Pharaoh_hound', 'ages/train/121.Plott', 'ages/train/122.Pointer', 'ages/train/123.Pomeranian', 'ages/train/124.Poodle', 'ages/train/125.Portuguese_water_dog', 'ages/train/126.Saint_bernard', 'ages/train/127.Silky_terrier', 'ages/train/128.Smooth_fox_terrier', 'ages/train/129.Tibetan_mastiff', 'ages/train/130.Welsh_springer_spaniel', 'ages/train/131.Wirehaired_pointing_griffon', 'ages/train/132.Xoloitzcuintli', 'ages/train/133.Yorkshire_terrier']
    # Set the default graph as otherwise the model won't work
    graph = tf.get_default_graph()



def face_detector_algo(img):
    # Take advantage of the face detector algorithm
    photo_image = np.array(img)
    gray = cv2.cvtColor(photo_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



def dog_algo(img, Data, photo):

    '''
    Process the prediction based on the provided image/photo.
    img - a loaded image already in a matrix format that can be inferred upon by the models
    Data - result dictionary that will store the algorithm response to be sent back to the querying party
    photo - a raw image that can be used by CV2 to detect faces
    '''
    #   Return the prediction as well as the probability that, in fact, the predicted breed is the one in the picture(chosen among
    #   other breeds)

    prediction, probability = best_resnet_predict_breed(img)
    #   If the face detector identifies a face, go with the follwoing

    if face_detector_algo(photo) == True:

        label = {"label": "This picture looks like a {}".format(prediction)}
        Data["predictions"] = label
        # Data["predictions"].append(label)
        return Data

    #   If thte face detector did not identify any faces in the image and the confidence that the given breed is in the picture
    #   is more than 24 percent, go ahead and provided the dog breed estimation

    elif probability > 0.24:
        label = {"label": "This picture shows a {} dog".format(prediction)}
        Data["predictions"] = label
        # Data["predictions"].value(label)

        return Data

    #   If no face is dected and the model's confidence is lower then the given confidence return an error messsage

    else:
        label = {"label": 'There is neither a dog nor a human in this photo!'}
        Data["predictions"] = label
        # Data["predictions"].append(label)
        return Data


def best_resnet_predict_breed(img_path):
    '''
    This function takes in an immage path and with the use of Resnet50 bottelneck features and a trained model returns
    a prediciton
    Input:

    img_path - path to an image that will be inferred upon

    Output:
    The function outputs a prediction what dog breed is presented on the picutre, as well as the confidence with which this
    bread was calculated by the model

    '''
    # use the default graph to infer
    with graph.as_default():
        # extract bottleneck features
        bottleneck_feature = model.predict(img_path)
        # obtain predicted vector
        predicted_vector = model_custom.predict(bottleneck_feature)

        # return dog breed that is predicted by the model

        print(np.max(predicted_vector))
        return dog_names[np.argmax(predicted_vector)].split('.', 2)[1], np.max(predicted_vector)

@app.route('/')
@app.route('/index')
def index():
    # Render the webpage, no fireworks needed here
    return render_template('master.html')


@app.route('/predict', methods=["POST"])
def predict():
    # A rest-api type webpage that will return dog bread classification for the image sent from the index page
    # initialize the data dictionary that will be returned from the

    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        if flask.request.files.get("file"):
            # Read the file that was sent in the request
            photo = flask.request.files["file"].read()
            photo = Image.open(io.BytesIO(photo))

            # Resize the image to fit the Resnet input parameters
            photo = photo.resize((224, 224))

            # Conver the image to an array and expand it to add a dimension needed by the models to query
            x = image.img_to_array(photo)
            tensor = np.expand_dims(x, axis=0)

            data["predictions"] = []
            # classify the input image and then initialize the list
            # of predictions to return to the client
            data = dog_algo(tensor, data, photo)

            # indicate that the request was a success
            data["success"] = True


    # return the data dictionary as a JSON response
    return flask.jsonify(data)
