#!pip install lime
#!pip install -U keras
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import Xception, DenseNet169, ResNet50, MobileNetV2
#from tensorflow.keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn import metrics
from time import time

pd.set_option('display.max_colwidth', -1)


def create_cnn(width, height, depth):
  inputShape = (height, width, depth)

  conv_base = MobileNetV2(weights='imagenet',include_top=False, input_shape=inputShape, pooling='avg')
  print('This is the number of trainable weights before freezing the conv base:', len(conv_base.trainable_weights))
  conv_base.trainable = False
  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.Flatten())
  #model.add(layers.Dense(256, activation='relu'))
  #model.add(layers.BatchNormalization())
  #model.add(layers.Dense(64, activation='relu'))
  #model.add(layers.BatchNormalization())
  model.add(layers.Dense(2, activation='softmax')) #linear

  print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
  model.summary()
  return model

df = pd.read_csv('data/set_1_2_3_1000.csv')[['image', 'score']]
#(train, test) = train_test_split(df, test_size=0.25, random_state=42)
#df['bin_score'] = df.score.apply(lambda x: to_categorical(0 if x < 7 else 1, num_classes=2))
df['bin_score'] = df.score.apply(lambda x: 'low' if x < 6. else 'high')
df.drop('score', axis=1, inplace=True)
#train, validate, test = np.split(df.sample(frac=1), [int(.80*len(df)), int(.9*len(df))])  #split to: train 80%, test 10%, validation 10%
df['image_name'] = df.image.apply(lambda x: x[5:11])
image_num = df.image_name.unique()
train, test, validate = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
train = df[df.image_name.isin(train)][['image', 'bin_score']].sample(frac=1)
test = df[df.image_name.isin(test)][['image', 'bin_score']].sample(frac=1)
validate = df[df.image_name.isin(validate)][['image', 'bin_score']].sample(frac=1)
print(train.head())
print(f'Train: {len(train)}, Test: {len(test)}, Validation:{len(validate)}')
print(f'Train.high: {len(train.loc[train.bin_score == "high"])}') #2169 low 2017 high.
print(f'Train.low: {len(train.loc[train.bin_score == "low"])}')

"""## Main model training."""

#Differentiable F1 loss.
def f1_loss(logits, labels):
  __small_value=1e-6
  beta = 1.
  batch_size = tf.dtypes.cast(tf.size(logits), tf.float32)
  p = tf.math.sigmoid(logits)
  l = labels
  num_pos = tf.math.reduce_sum(p) + __small_value
  num_pos_hat = tf.math.reduce_sum(l) + __small_value
  tp = tf.math.reduce_sum(l * p)
  precise = tp / num_pos
  recall = tp / num_pos_hat
  fs = (1. + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
  loss = tf.math.reduce_sum(fs) / batch_size
  return (1. - loss)

input_x = input_y = 1000
directory = "data/"
train_datagen = ImageDataGenerator(
      rescale=1./255,
      samplewise_center=True,
      samplewise_std_normalization=True,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      #brightness_range = [0.5, 1.5],
      shear_range=0.2,
      #zoom_range=0.2,
      #horizontal_flip=True,
      #channel_shift_range=10.,
      #vertical_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=directory,
                                                    x_col="image",
                                                    y_col="bin_score",
                                                    #class_mode="raw",
                                                    class_mode='categorical',
                                                    target_size=(input_y, input_x),
                                                    batch_size=16)

test_datagen = ImageDataGenerator(
      rescale=1./255,
      samplewise_center=True,
      samplewise_std_normalization=True
      )
test_generator = test_datagen.flow_from_dataframe(dataframe=test,
                                                  directory=directory,
                                                  x_col='image',
                                                  y_col='bin_score',
                                                  #class_mode="raw",
                                                  class_mode='categorical',
                                                  target_size=(input_y, input_x),
                                                  batch_size=16)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model_{val_auc:03f}.h5', save_best_only=True, monitor='val_auc', mode='max', verbose=1) #{epoch:03d}-{auc:03f}-{val_auc:03f}
reduce_lr_loss = ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=7, verbose=1, min_delta=1e-6, mode='max')

model = create_cnn(input_y, input_x, 3)
#model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=2e-5), metrics=['mae']) #RMSprop

#model = load_model('model.h5')
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adamax(lr=2e-4), metrics=['AUC']) #categorical_hinge f1_loss

model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[mcp_save, reduce_lr_loss]) #validation_steps=20

"""
##Validate.

validate = df
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(dataframe=validate,
                                                              directory='/tmp/TIS/',
                                                              x_col='image',
                                                              y_col='bin_score',
                                                              class_mode='categorical', #None
                                                              target_size=(250, 250),
                                                              batch_size=64)


prediction = model.predict(validation_generator,
                                     steps=None,
                                     callbacks=None,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False,
                                     verbose=0)
validation_y = validate['bin_score'].values
scores = model.evaluate(validation_generator)
#print(model.metrics_names)
#print(scores)
#p = [x[0] for x in prediction]
#MSE = metrics.mean_squared_error(p,validation_y)
#print("Final score (MSE): {}".format(MSE))
#MAE = metrics.mean_absolute_error(p, validation_y)
#print("Final score (MAE): {}".format(MAE))
#R2 = metrics.r2_score(p,validation_y)
#print("Final score (R2): {}".format(R2), ' ', np.corrcoef(validation_y, p))
"""

"""
##Explain.

img_list = list(train.head(40).image.values)
out = []
out_ind = []
for ind, img_name in enumerate(img_list):
    img = image.load_img('/tmp/TIS/all_tagged_tejal/'+img_name, target_size=(250, 250))
    x = image.img_to_array(img)
    #plt.figure()
    x = np.expand_dims(x, axis=0)
    print(model.predict(x)[0], 'high' if model.predict(x)[0][1] > 0.7 else 'low', train.iloc[ind]['bin_score'])
    if (model.predict_classes(x)[0] == 1) and (train.iloc[ind]['bin_score'] == 'high'):
    #x = Xception.preprocess_input(x)
      out_ind.append(ind)
    out.append(x)
out = np.vstack(out)
preds = model.predict(out)
plt.imshow(out[0]/255.)

from lime import lime_image

for i in out_ind: #np.where(preds[:,0] < preds[:,1])[0]:
  explainer = lime_image.LimeImageExplainer()
  explanation = explainer.explain_instance(out[i], model.predict, hide_color=0, num_samples=1000)
  #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
  plt.figure()
  plt.imshow(mark_boundaries(temp / 255., mask))
  """
