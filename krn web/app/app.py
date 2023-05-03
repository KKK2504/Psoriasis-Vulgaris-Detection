from flask import Flask, render_template , request
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def hello_world():
  if request.method == 'POST':
    file = request.files['file']
    img_bytes = file.read()

    # Convert the uploaded image to a cv2 image
    nparr = np.fromstring(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    print("prediction",prediction)
    confidence_score = prediction[0][index]
    print("Class index",index)
    print("This Disease is :",class_name)
    print("Accuracy",confidence_score)
    return render_template('index.html',class_name=class_name, confidence_score=confidence_score)
  else:
    return render_template('index.html')