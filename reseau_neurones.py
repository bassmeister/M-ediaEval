import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
def data_split(dirc):
	for root, dirs, files in os.walk(dirc):
		for name in files:
			if name.endswith((".jpg")):
				try:
			#        for j in range(len(os.listdir(path+listvid['files'][i]))):
					
					impath = root+'/'+name
	#                print(impath)
			#            print(impath)
			#            im = img_fltr(impath,32)
					print(impath)
					image = Image.open(impath)
					image.thumbnail((150,150))
					neur='neur'
					image.save(impath[:-4] + 'neur.jpg')
					image.close
					os.remove(impath)
				except:
					print(name)
 data_split(validation_data_dir)
 data_split(training_data_dir)
        
#Data augmentation		
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
# width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
# rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
# shear_range is for randomly applying shearing transformations
# zoom_range is for randomly zooming inside pictures
# horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
# fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=rootdir, save_prefix='test', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
        
#now that the data augmentation funcion has been tested we can start training the dataset
rootdir = 'C:/Users/Amstrong/Documents/MediaEval/database/'
train_data_dir = rootdir +'trainee'
validation_data_dir = rootdir +'testee'
nb_train_samples = 5327
nb_validation_samples = 1777
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Reshape
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

model.add(Conv2D(32, (3, 3),dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescalingi :! 
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=5326 // 24,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=1778 // 24)
model.save_weights(rootdir + 'first_try.h5')
