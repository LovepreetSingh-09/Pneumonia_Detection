
# Basic Libraries
import numpy as np
from PIL import Image
from glob import iglob, glob
import os
import matplotlib.pyplot as plt

# Keras from Tensorflow for building Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

# For Evaluation 
from sklearn.metrics import confusion_matrix, accuracy_score

tf.__version__

#tf.test.is_gpu_available()
tf.test.is_built_with_cuda()

k.clear_session()

os.getcwd()
path = 'chest-xray-pneumonia/'
train_files = glob(os.path.join(path, 'train/', '*/*.jpeg'))
len(train_files)
test_files = glob(os.path.join(path, 'test', '*/*.jpeg'))
len(test_files)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, rotation_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_it = train_datagen.flow_from_directory('chest-xray-pneumonia/train/',
                        target_size=(156,156), color_mode='rgb',batch_size=16)

train_it.target_size
train_it.class_indices
train_it.color_mode


test_it = test_datagen.flow_from_directory(
    'chest-xray-pneumonia/test/', target_size=(156, 156), color_mode='rgb', shuffle=True,batch_size=16)

batch_x, batch_y = test_it.next()
batch_x.min()
batch_x.max()
batch_x.shape
batch_y

labels_2_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
idx_2_labels = {0:'NORMAL', 1:'PNEUMONIA'}


''' Manually importing images data with labels
 ================================================================================'''
label_categories=['NORMAL','PNEUMONIA']
data=dict()
labels={}
size=(156,156)
train_lab,test_lab=[],[]
for a in ['test']:
    files,lab=[],[]
    for i,cat in enumerate(label_categories):
        for f in iglob(os.path.join(path,a,cat,'*.jpeg')):
            files.append(np.array(Image.open(f).resize(size,Image.ANTIALIAS).convert('RGB')))
            lab.append(i)          
    data[a]=np.array(files)
    labels[a]=np.array(lab)
    
# data['train'].shape    
data['test'].shape    
data['test'][1].shape
#x_train,y_train=data['train']/255.,labels['train']
#x_train.shape, y_train_.shape 
#
#labels_no,count=np.unique(y_train,return_counts=True)    
#labels_counts=dict(zip(labels_no,count))
#labels_counts
x_test,y_test_=data['test']/255.,labels['test']
rand_idx=np.random.randint(0,len(x_test),size=len(x_test))
y_test_=y_test_[rand_idx]; y_test_
x_test=x_test[rand_idx]
np.array(x_test).shape
y_test_=np.array(y_test_)
#y_train=to_categorical(y_train); y_train.shape
y_test=to_categorical(y_test_)



fig, axes = plt.subplots(4, 2, figsize=(12, 20), subplot_kw={
                         'xticks': (), 'yticks': ()})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(batch_x[i],cmap='gray')
    ax.set_title(idx_2_labels[np.argmax(batch_y[i])])
plt.tight_layout()

idx_2_labels[1]
image_shape = batch_x.shape[1:]
image_shape
   
def train_generator(train_it,valid_it,save_dir):
    idx_2_labels[1]
    image_shape = batch_x.shape[1:]
    image_shape
    
    num_classes = 2
    
    model_name = 'model2-{epoch:03d}-{val_accuracy:.2f}.h5'
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, model_name)
    
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=2,
                                 monitor='val_loss',min_lr=0.5e-06,verbose=1)
    callbacks = [lr_reducer,checkpoint]
    
    base_model = ResNet50V2(weights='imagenet', include_top=False,
                          input_shape=image_shape)
    
    base_model.summary()
    base_model.trainable=False
    #for layer in base_model.layers[:200]:
    #    layer.trainable=False
    
    x=base_model.output
    x=Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.summary()
    optimizer=Adam(lr=0.00001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    
    
    model.fit_generator(train_it,validation_data=valid_it,epochs=10,callbacks=callbacks)


save_dir = os.path.join(os.getcwd(), 'saved_models')
save_dir

train_generator(train_it,test_it,save_dir)

model_loc='model2-006-0.89.h5'
m=load_model(os.path.join(save_dir,model_loc))

m.summary()
loss,acc=m.evaluate(x_test,y_test,batch_size=10)
loss
acc

y_pred=m.predict(x_test)
idx_pred=np.argmax(y_pred[0:10],axis=1)
prob=np.max(y_pred[0:5],axis=1)
for i in range(len(idx_pred)):
    print(idx_2_labels[idx_pred[i]])

y_pred=np.argmax(y_pred,axis=1)
pred_class,actual_class=[],[]
for i in range(len(y_pred)):
    pred_class.append(idx_2_labels[y_pred[i]])
    actual_class.append(idx_2_labels[y_test_[i]])

acc=accuracy_score(pred_class,actual_class)
print('\nAccuracy_Score....\n\t{} %'.format(np.round(acc*100,4)))

cm=confusion_matrix(actual_class,pred_class)
print('\nConfusion_matrix.....\n',cm)

precision=cm[0,0]/(cm[0,0]+cm[1,0])
recall=cm[0,0]/(cm[0,0]+cm[0,1])
print('\nPrecsion_Score ....\n\t',np.round(precision,4))
print('\nRecall_Score.....\n\t',np.round(recall,4))

f1=2*precision*recall/(precision+recall)
print('\nF1_Score.....\n\t',np.round(f1,4))

y=x_test[0].reshape(1,156,156,3)
print(m.predict(y))



