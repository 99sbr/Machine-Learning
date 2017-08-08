import numpy as np
import glob
from scipy import misc
train = glob.glob("/home/delhivery/Desktop/images/train/*.jpg")
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
dic = {"iphone":0,"samsung":1,"lenovo":2,"watch":3,"others":4}
for f in train:
    label = str(f.split("_")[3])
    image = misc.imread(f).astype(np.uint8)
    if image.shape == (150, 150, 3):
        x_train.append(image)
        y_train.append(dic[label])


x_train = np.array(x_train)
y_train = np.array(y_train)


print("x_train shape :",x_train.shape)
print("y_train shape :",y_train.shape)
