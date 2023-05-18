from mnist_mlp_cpu import MNIST_MLP
import numpy as np
from PIL import Image
import cv2
import gradio as gr

def build_mnist_mlp(param_dir='checkpoint/mlp-784-16-1epoch.npy'):
    h1, h2, e = 784, 16, 1
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.build_model()
    mlp.init_model()
    mlp.load_model(param_dir)
    return mlp


def predict(img):
    mlp = build_mnist_mlp()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    new_size = (28, 28) # 新的图像大小
    img = cv2.resize(img, new_size)
    img = np.array(img).flatten()
    out = mlp.forward(img)
    label = np.argmax(out)
    return "预测的结果为: " + str(label)

# 定义 Gradio 接口
demo = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(None, None)),
    outputs=gr.outputs.Textbox(label="Predicted digit"),
    title="MNIST Handwritten Digit Recognition",
    description="Upload an image of a handwritten digit and the model will predict the digit.",
)
demo.launch(share=True)
 