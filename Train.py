
import numpy as np
import pandas as pd
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/data/fer2013/fer2013.csv')
df.shape

unique_usages = df['Usage'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=unique_usages.index, y=unique_usages.values, palette="viridis")
plt.title('Biểu đồ phân bổ các mẫu test')
plt.xlabel('Loại')
plt.ylabel('Số lượng')
plt.show()

train_set = df[df['Usage'] == 'Training']
test_set = df[df['Usage'] == 'PrivateTest']


print("Thông tin tệp huấn luyện")
print(train_set.info())

print("\n Thông tin tệp kiểm tra")
print(test_set.info())


plt.figure(figsize=(8, 6))
sns.countplot(x='emotion', data=df)
plt.title('Phân phối cảm xúc')
plt.show()

pixels = df['pixels'].tolist()
X = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    X.append(face.astype('float32') / 255.0)

X = np.array(X)
X = np.expand_dims(X, -1)
y = df['emotion'].values


def preprocess_data(data):
    pixel_lists = df['pixels'].tolist()
    images = []

    for pixel_sequence in pixel_lists:
        pixel_values = [int(pixel) for pixel in pixel_sequence.split(' ')]
        image = np.asarray(pixel_values).reshape(48, 48)
        images.append(image.astype('float32') / 255.0)

    X = np.array(images)

    X = np.expand_dims(X, -1)

    y = df['emotion'].values

    return X, y


X_train, y_train = preprocess_data(train_set)
X_test, y_test = preprocess_data(test_set)