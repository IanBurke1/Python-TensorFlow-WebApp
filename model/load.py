import keras.models
from keras.models import model_from_json
import tensorflow as tf

#
#
#


def loadModel():
    json_file = open('mnistModel.json','r') # open json file
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    # Load weights into new model
    model.load_weights('mnistModel.h5')
    # compile and evaluate loaded model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.get_default_graph()

    return model, graph
    