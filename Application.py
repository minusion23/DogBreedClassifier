from DogClassifierApp import app
from DogClassifierApp.routes import load_models
# Load the models at the time the server is set up, so that the models are ready for querying
load_models()
app.run(host='127.0.0.1', port=3001, debug=True)
