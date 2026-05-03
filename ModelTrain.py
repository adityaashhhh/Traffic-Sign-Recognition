

import numpy as np           
import pandas as pd          
import os    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential        # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import to_categorical    # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Reshape, MaxPooling2D, Flatten, Dropout, BatchNormalization      # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping        # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.preprocessing.image import ImageDataGenerator      # type: ignore
import warnings
warnings.filterwarnings('ignore')

# !!!! MAKING IT TABULAR!!!!


base_path = r'C:\VIsemMinor\TSRS'

train_df = pd.read_csv(os.path.join(base_path, "Train.csv"))
test_df  = pd.read_csv(os.path.join(base_path, "Test.csv"))
meta_df  = pd.read_csv(os.path.join(base_path, "Meta.csv"))

train_dir = os.path.join(base_path, "Train")
test_dir  = os.path.join(base_path, "Test")
meta_dir  = os.path.join(base_path, "Meta")

label_cod={0:'Speed limit (20km/h)',1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
           3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
           6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
           9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
           12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 
           17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 
           21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
           25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
           29:'Bicycles crossing', 30:'Beware of ice/snow',31:'Wild animals crossing', 
           32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 
           36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 
           40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

#TRAIN

labels = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '5', '6', '7', '8', '9']

img_list1 = []
label_list1 = []

for label in labels:
    for img_file in os.listdir(train_dir + "/" + label):
        img_list1.append(train_dir + "/" + label + "/" + img_file)
        label_list1.append(int(label))

train_df=pd.DataFrame({'img':img_list1,'label':label_list1})

train_df['encode_label']=train_df['label'].map(label_cod)

#print(train_df.sample(5))

#print(os.listdir(test_dir))


#TEST

ttest_df = pd.read_csv(os.path.join(base_path, "Test.csv"))


test_img_list = [os.path.join(base_path, path) for path in ttest_df['Path']]


test_df = pd.DataFrame({'img': test_img_list,'label': ttest_df['ClassId']})

test_df['encode_label'] = test_df['label'].map(label_cod)

#print(test_df.sample(5))

#META


meta_files = [f for f in os.listdir(meta_dir) if f.endswith('.png')]
meta_df = pd.DataFrame({'img': [os.path.join(meta_dir, f) for f in meta_files],'label': [int(f.split('.')[0]) for f in meta_files]})
meta_df['encode_label'] = meta_df['label'].map(label_cod)

#print(meta_df.sample(5))

import seaborn as sns


# plt.figure(figsize=(12, 6))
# sns.histplot(x=train_df['encode_label'], label='train')
# sns.histplot(x=test_df['encode_label'], label='test')
# sns.histplot(x=meta_df['encode_label'], label='meta')
# plt.xticks(rotation=90) 
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.title('Label Distribution')
# plt.legend()


#plt.show()


# sample_df = train_df.drop_duplicates('label').sort_values('label')

# plt.figure(figsize=(15, 12))
# for i, row in enumerate(sample_df.values):
#     plt.subplot(7, 7, i+1) # 43 sınıf için 7x7 grid yeterli
#     plt.imshow(plt.imread(row[0])) # row[0] resim yolu (img)
#     plt.title(row[1], fontsize=8) # row[1] etiket (label)
#     plt.axis('off')

# plt.tight_layout()
# plt.show()



# Data Processing


df = pd.concat([train_df, test_df, meta_df], ignore_index=True)

df.info()

df.sample(10)

# #Normalization

x = []
for img_path in df['img']: 
     img=cv2.imread(str(img_path))
     if img is None:
         print(f"Resim yüklenemedi: {img_path}") 
         continue
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we convert RGB for the mobile app
     img=cv2.resize(img, (30, 30))
     img=img / 255.0
     x.append(img)


x = np.array(x)


# x.shape 


y=df[['label']]

x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42, test_size=0.20)

y_train = np.array(y_train)
y_test = np.array(y_test)

#model training

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# model = Sequential([
#     Input(shape=(30, 30, 3)),
    
#     Conv2D(32, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(32, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
#     Dropout(0.2),

#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
#     Dropout(0.3),

#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
#     Dropout(0.4),

#     Flatten(),
#     Dense(512, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(43, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# early_stop=EarlyStopping(monitor='val_loss',patience=10)
# history=model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=20, callbacks=[early_stop], verbose=1)


# model.save('traffic_sign.h5')


# Model Sowing

model = tf.keras.models.load_model("traffic_sign.h5")

num_samples = len(x_test)
random_indices = np.random.choice(num_samples, size=min(43, len(x_test)), replace=False)

x_test_resized = x_test[random_indices]
y_test_array = np.array(y_test)
actual_labels = y_test_array[random_indices].flatten()

predictions = model.predict(x_test_resized)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(7, 7, i + 1)
    plt.imshow(x_test[idx])

    actual_text = label_cod[actual_labels[i]]
    predicted_text = label_cod[predicted_labels[i]]
    
    plt.title(f"Actual: {actual_text}\nPredicted: {predicted_text}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

#!!!!   IMPORTANTTTT        !!!!!


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix - Traffic Signs')

plt.show()


from sklearn.metrics import accuracy_score
print(f"Test set accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")

import pickle
import tensorflow as tf

# # Save keras model
# model.save("traffic_model.keras")

# # Save labels
# with open('labels.pkl', 'wb') as f:
#     pickle.dump(label_cod, f)

# # Convert to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('traffic_sign_model.tflite', 'wb') as f:
#     f.write(tflite_model)

# print("Process complete: files saved.")