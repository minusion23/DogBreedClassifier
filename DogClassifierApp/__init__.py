from flask import Flask
# Import the routes and start Flask run
app = Flask(__name__)

from DogClassifierApp import routes
