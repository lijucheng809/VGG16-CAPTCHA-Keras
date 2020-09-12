from keras.callbacks import ModelCheckpoint,TensorBoard
import getConfig
from cnnModel import CNN
from Image import Image

gConfig=getConfig.get_config(config_file='config.ini')

def train():
    model = CNN(gConfig['rate'])
    model = model.creatModel()
    check_point = ModelCheckpoint(
        filepath = './model_dir/train.h5',
        save_best_only = True
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    image = Image()
    model.fit_generator(
        image.gen(),
        steps_per_epoch=10000, #每次迭代需要多少个Batch
        epochs=5,              #共需迭代几次
        validation_data= image.gen(),
        validation_steps=10,
        callbacks=[check_point])

if __name__=='__main__':
    train()