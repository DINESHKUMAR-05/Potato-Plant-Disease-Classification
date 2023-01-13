import tensorflow
from tensorflow import keras
import gradio as gr

model = keras.models.load_model('simple_model.h5')
potato_classes = ['Early_Blight','Healthy','Late_Blight']

def predict_input_image(img):
  img_3d=img.reshape(-1,256,256,3)
  prediction=model.predict(img_3d)[0]
  return {potato_classes[i]: float(prediction[i]) for i in range(3)}

image = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label(num_top_classes=3)
gr.Interface(fn=predict_input_image, inputs=image, outputs=label,interpretation='default').launch()