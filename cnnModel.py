from keras.layers import Input,MaxPooling2D,Dense,BatchNormalization,Flatten,Conv2D,Dropout
from keras.models import Model
import getConfig

gConfig=getConfig.get_config(config_file='config.ini')
class CNN:
    def __init__(self,rate):
        self.rate = rate
    def creatModel(self):
        h, w, nclass = gConfig['hight'] , gConfig['width'] , gConfig['num_datasetclasses']
        input_tensor = Input(shape=(h, w, 3))
        x = input_tensor
        for i in range(4):
            x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization(axis=3)(x)  # 3指的是图片的通道数量
            x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dropout(self.rate)(x)
        x = [Dense(nclass, activation='softmax', name='D%d' % (n + 1))(x) for n in range(4)]
        model = Model(inputs=input_tensor, outputs=x)
        return model
