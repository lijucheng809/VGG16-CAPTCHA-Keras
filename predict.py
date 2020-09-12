from captcha.image import  ImageCaptcha
import string,random
import numpy as np
from keras.layers import Input,MaxPooling2D,Dense,BatchNormalization,Flatten,Conv2D,Dropout
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import getConfig

gConfig=getConfig.get_config(config_file='config.ini')
def randomCode():
    raw=string.digits+string.ascii_uppercase
    code=''.join(random.sample(raw,4))
    return code,raw

def gen(height=gConfig['hight'],width=gConfig['width'],batch_size=1,nclass=gConfig['num_datasetclasses']):
    X=np.zeros((batch_size,height,width,3),dtype=np.float)
    y=[np.zeros((batch_size,nclass),dtype='uint8') for i in range(4)]
    generator=ImageCaptcha(height=height,width=width)
    while True:
        for i in range(batch_size):
            random_str,raw=randomCode()
            genCode=generator.generate_image(random_str)
            X[i]=np.array(genCode).astype('float32')/255.
            for j,ch in enumerate(random_str):
                y[j][i,:]=0
                y[j][i,raw.find(ch)]=1
        yield X,y,raw,genCode,random_str

def decode(y,raw):
    y=np.argmax(np.array(y),axis=2)[:,0]
    return ''.join([raw[i] for i in y])

#预测
model=load_model('model_dir/train.h5')
correct=0

for i in range(1000):
    X,y,raw,genCode,random_str = next(gen())
    result=model.predict(X)
    decode_result=decode(y=result,raw=raw)
    #print(random_str)
    #print(decode_result)
    if random_str==decode_result:correct+=1
    print("totoal is",i+1," correct is",correct)

