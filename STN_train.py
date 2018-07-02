#-----Dependencies-----#


import numpy as np
import keras
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD
import keras.backend as K
from spatial_transformer import SpatialTransformer
import cv2
import random

#-----Define global variables-----#
batch_size = 64
photo_width = 320
photo_height = 240

#-----DATA PREPROCESSING----#


#-----Just set to 0 each value under the threshold-----#
def threshhold_data(x, thresh, value):
    im = x
    im[im > thresh] = value
    return im



#-----Some utility methods for data augmentation-----#

#-----Makes a random rotation to the image from -30 to 30 degrees-----#
def random_rotation(x):
    theta = random.randrange(-30,30)
    rows, cols, _ = x.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(x,M,(cols,rows))
    return dst


def data_augmentation(x):
    from keras import preprocessing
    x = x.reshape((photo_height, photo_width, 1))
    x = preprocessing.image.random_zoom(x, [0.90, 1.1])
    #x = preprocessing.image.random_shear(x, 0.05)
    x = preprocessing.image.random_shift(x, 0.2, 0.2)
    x = random_rotation(x)
    return x


#-----Read the dataset for training. Add a target label for each train sample-----#
def read_train_set(folderDatasetPath):
    import os
    
    x_set = []
    y_set = []
    for root, dirs, files in os.walk(folderDatasetPath, topdown=False): 
        n = 0
        for name in files:
            
            file = os.path.join(root, name)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                #-----Find the relative target for this folder-----#
                if file.lower().endswith('target.png'):
                    y_r = cv2.imread(file, -1)
                    y = cv2.resize(y_r, dsize=(photo_width,photo_height), interpolation=cv2.INTER_NEAREST)
                    print file
                    
                    thresh = np.min(y)+300
                    y = threshhold_data(y, thresh, thresh)   
                    y = (y-np.min(y))/300.            
                    y_set.append(y)
        
                else:
                    
                    x_r = cv2.imread(file, -1)
                    x = cv2.resize(x_r, dsize=(photo_width,photo_height), interpolation=cv2.INTER_NEAREST)
                    
                    thresh = np.min(x)+300
                    
                    x = threshhold_data(x, thresh, thresh)

                    x = (x-np.min(x))/300.
                    x_set.append(x)
                    #for i in range(2):
                        #x_set.append(data_augmentation(x))
                    n += 1
                    
                    
        #-----Now we add the target label for each train sample-----#
        for i in range(n-1):
            if y_set:
                y_set.append(y_set[len(y_set)-1])
    
    
        
    #---Prepare the data for fitting, normalizing in [0,1]. To the background is associated 0 (black), to the foreground 1---#
    #---We normalize respect to the max value of intensity (depth) of the whole dataset to preserve the depth information in each photo---#

    X = np.array(x_set).reshape((-1, photo_height, photo_width, 1)).astype('float32')
    y = np.array(y_set).reshape((-1, photo_height, photo_width, 1)).astype('float32')
    
    X = 1-X
    
    
    y = 1-y
    
    #----Prepare the validation set----#
    n_val_examples = (int)(len(X)/10)
    val_indexes = random.sample([i for i in range(len(X))], n_val_examples)
    X_val = X[val_indexes]
    y_val = y[val_indexes]
    X = np.delete(X, val_indexes, axis=0)
    y = np.delete(y, val_indexes, axis=0)
    return X, y, X_val, y_val






#-----Define the callback to save training images at the beginning of each epoch overriding the 'on_epoch_begin' method-----#

from keras.callbacks import Callback
class viewer(Callback):
    
    
    def __init__(self, todraw_sample_index):
        self.index = todraw_sample_index
        
        
    #-----The following two methods are for the grid visualization-----#
    def create_point_lines(self, grid, width, height):
        grid = np.reshape(grid, (3, -1))[0:2] 

        tmp_horizontal = []
        for i in range(height):
            tmp_horizontal.append(grid[:, i*(width):(i+1)*(width)].T)
        tmp_horizontal = np.array(tmp_horizontal)

        tmp_vertical = []
        for i in range(width):
            tmp_vertical.append(tmp_horizontal[0:height,i])
        tmp_vertical = np.array(tmp_vertical)

        return tmp_horizontal, tmp_vertical

    
    def draw_point_lines(self, grid, width, height, draw):

        tmp_h, tmp_v = self.create_point_lines(grid, width, height)

        #draw points (some points will be draw more times)
        tmp_p = np.reshape(tmp_h, (-1, 2))
        for i in range(len(tmp_p)):
            x, y = tmp_p[i][0], tmp_p[i][1]
            draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill = (255,0,0))

        #draw lines
        for i in range(len(tmp_h)):
            for j in range(len(tmp_h[i])-1):
                x0, y0, x1, y1 = tmp_h[i][j][0], tmp_h[i][j][1], tmp_h[i][j+1][0], tmp_h[i][j+1][1]
                draw.line([x0, y0, x1, y1], fill = (255,179,102))
        for i in range(len(tmp_v)):
            for j in range(len(tmp_v[i])-1):
                x0, y0, x1, y1 = tmp_v[i][j][0], tmp_v[i][j][1], tmp_v[i][j+1][0], tmp_v[i][j+1][1]
                draw.line([x0, y0, x1, y1], fill = (255,179,102))
    
    
    #-----This returns both output of STN (the transformed image) and the parameters of the transformation matrix-----#
    def get_transformation_params(self):
        
        #------STN MODULE TRANSFORMATION------#
        X1 = self.model.input
        Y1 = self.model.layers[0].output
        stn = K.function([X1], [Y1])

        #------WEIGHTS OF LOCALISATION NET------#
        init = self.model.layers[0].locnet.input
        output = self.model.layers[0].locnet.output
        theta = K.function([init], [output])
        
        
        test = X_val[self.index].reshape((1,photo_height,photo_width, 1))
        res1 = np.squeeze(np.array(stn([test])))
        transf_params = np.squeeze(np.array(theta([test])))
        return transf_params, res1
    
    
    def on_epoch_begin(self, epoch, logs= None):
        
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib import cm
        
        #-----GET PARAMS AND PREDICTION FOR THE TRAIN IMAGE-----#
        transf_params, res1 = self.get_transformation_params()
        
        
        #-----MODULE OUTPUT-----#
        res1 = np.array(res1)
        res1 = np.squeeze(res1)
        im1 = res1
        
        #-----BUILD THE CANVAS-----#
        canvas = Image.new(mode = 'RGB', size = (1160, 340), color = (255,255,255))
        canvas.paste(Image.fromarray(cm.gray(X_val[self.index].reshape((photo_height,photo_width)), bytes=True)), (50, 50))
        canvas.paste(Image.fromarray(cm.gray(y_val[self.index].reshape((photo_height,photo_width)), bytes=True)), (photo_width+100, 50))
        canvas.paste(Image.fromarray(cm.gray(im1.reshape((photo_height,photo_width)), bytes=True)), (2*photo_width+150, 50))
        

        draw = ImageDraw.Draw(canvas)

        width = 4
        height = 4
        
        #-----Want a regular grid in [-1,1]-----#
        x_ = np.linspace(-1,1,width)
        y_ = np.linspace(-1,1,height)

        x_c, y_c = np.meshgrid(x_, y_)
        x_c = np.reshape(x_c, [-1])
        y_c = np.reshape(y_c, [-1])
        ones = np.ones_like(x_c)
        indices_grid = np.concatenate([x_c, y_c, ones], 0).reshape((3, 16))
        
        
        R = transf_params.reshape((3,3))
        
        indices_grid = indices_grid.reshape((3,-1))
        
        
        #-----Compute the transformed grid from a regular one in [-1,1]-----#
        grid_rot = np.matmul(R, indices_grid)

        
        #------Scale the transformed grid in the correct dimension for canvas------#
        #---Transformation maps to a region on canvas of dimension [photo_width, photo_height] 
    
        
        # Define scale and translation matrix
        x1 = np.min(grid_rot[0])
        x2 = np.max(grid_rot[0])
        y1 = np.min(grid_rot[1])
        y2 = np.max(grid_rot[1])
        
        #-----For general input interval in the transformation-----#
        l = x2-x1
        h = y2-y1
        
        #-----For input interval equal to [-1,1]-----#
        #l = 2
        #h = 2
        
        pos_x = 50
        pos_y = 50
        S = np.array([[photo_width/l, 0, -photo_width*x1/l + pos_x], [0, photo_height/h, -photo_height*y1/h + pos_y], [0, 0, 1]])
        drawable_grid_rot = np.matmul(S,grid_rot)

        # Draw on Canvas
        self.draw_point_lines(drawable_grid_rot, width, height, draw)
        
        print ("\nSaving transformation image...")

	myfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14, encoding="unic")
	myTitlefont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14, encoding="unic")

        draw.text((210 + photo_width, 30 + photo_height + 30), "Epoch number: " + str(epoch), fill='blue', font=myTitlefont)
        draw.text((160,30), "Train sample", fill='blue', font=myfont)
        draw.text((220+photo_width,30), "Target", fill='blue', font=myfont)
        draw.text((260+2*photo_width,30), "STN prediction: " + str(epoch), fill='blue', font=myfont)
        canvas.save("epoch_images/depth_"+str(epoch)+".png")





#-------BUILDING THE NETWORK-------#

input_shape = (photo_height, photo_width, 1)

# initial weights for last layer of localisation
#----At the beginning, we start with the identity transformation----#
b = np.zeros((3, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
b[2, 2] = 1
W = np.zeros((int(photo_width/8)*int(photo_height/8)*20, 9), dtype='float32')
#W = np.zeros((19980, 9), dtype='float32')
weights = [W, b.flatten()]


#-----We use just linear activation (identity function) to avoid non linear behavior of the CNN----#
#-----This is recommended for STN tasks-----#

#-----First define the Localisation Network which will output 9 parameters for the projective transformation-----#
locnet = Sequential()
locnet.add(Convolution2D(20, (3, 3), input_shape = input_shape, padding='same'))
locnet.add(Activation('linear'))
locnet.add(MaxPooling2D(pool_size=(2,2)))
locnet.add(Convolution2D(20, (3, 3), padding='same' ))
locnet.add(Activation('linear'))
locnet.add(MaxPooling2D(pool_size=(2,2)))
locnet.add(Convolution2D(20, (3, 3), padding='same'))
locnet.add(Activation('linear'))
locnet.add(MaxPooling2D(pool_size=(2,2)))

locnet.add(Flatten())
#locnet.add(Dense(100))
locnet.add(Activation('linear'))
locnet.add(Dense(9, weights=weights))
locnet.add(Activation('linear'))


#-----Now we add a STN layer which will output the transformed image-----#
transf1 = Sequential()
transf1.add(SpatialTransformer(localization_net=locnet, output_size=(photo_height, photo_width, 1), input_shape=input_shape))
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
transf1.compile(loss='mse', optimizer=sgd, metrics=['mse'])
locnet.summary()
transf1.summary()






#-----LOAD DATASET-----#
X, y, X_val, y_val= read_train_set("dataset/train_set")
print "Train tensor shape: ", X.shape
print "Validation tensor shape: ", X_val.shape





#-----FITTING-----#

#-----Fit on batch. This is useful for GPU training to prevent Out Of Memory (OOM) error-----#

def fgenerator(features, labels, batch_size):
    #----The generator for the train set----#
    while True:
     # Create empty arrays to contain batch of features and labels #
        batch_features = np.zeros((batch_size, photo_height, photo_width, 1))
        batch_labels = np.zeros((batch_size, photo_height, photo_width, 1))
        for i in range(batch_size):
        # choose random index in features
            index = random.randrange(features.shape[0])
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

        
def vgenerator(features, labels, batch_size):
    #----The generator for the validation set----#
    while True:
     # Create empty arrays to contain batch of features and labels #
        batch_features = np.zeros((batch_size, photo_height, photo_width, 1))
        batch_labels = np.zeros((batch_size, photo_height, photo_width, 1))
        for i in range(batch_size):
        # choose random index in features
            index = random.randrange(features.shape[0])
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels
        
        


#a = raw_input("Premi per continuare")
epoch_viewer = viewer(random.randrange(len(X_val)))
weights_saver = keras.callbacks.ModelCheckpoint("weights.h5", monitor='val_loss', verbose=0, save_best_only=True, \
                                                save_weights_only=True, mode='auto', period=1)
history = transf1.fit_generator(generator = fgenerator(X,y, batch_size), steps_per_epoch = len(X)/batch_size, \
        epochs = 200, shuffle = True, validation_data = vgenerator(X_val, y_val, batch_size),\
        validation_steps = len(X_val)/batch_size, callbacks=[epoch_viewer, weights_saver])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('graph.png')
