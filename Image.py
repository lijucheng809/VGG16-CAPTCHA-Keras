from captcha.image import  ImageCaptcha
import string,random
import numpy as np
import getConfig

gConfig=getConfig.get_config(config_file='config.ini')
class Image:
    def __init__(self):
        pass
    def randomCode(self):
        raw = string.digits + string.ascii_uppercase
        code = ''.join(random.sample(raw, 4))
        return code, raw
    def gen(self,height=gConfig['hight'], width=gConfig['width'], batch_size=gConfig['batch_size'], nclass=gConfig['num_datasetclasses']):
        X = np.zeros((batch_size, height, width, 3), dtype=np.float)
        y = [np.zeros((batch_size, nclass), dtype='uint8') for i in range(4)]
        generator = ImageCaptcha(height=height, width=width)
        while True:
            for i in range(batch_size):
                random_str, raw = self.randomCode()
                X[i] = np.array(generator.generate_image(random_str)).astype('float32') / 255.
                for j, ch in enumerate(random_str):
                    y[j][i, :] = 0
                    y[j][i, raw.find(ch)] = 1
            yield X, y