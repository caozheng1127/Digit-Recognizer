import pandas as pd
import matplotlib.pyplot as plt, matplotlib as maping
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Loading the data
labeled_image = pd.read_csv('E:/Kaggle_Data/Digit Recognizer/train.csv')
images = labeled_image.iloc[:,1:]
labels = labeled_image.iloc[:,:1]
train_images,test_images,train_labels,test_labels = train_test_split(images, labels, train_size=0.8)

# Viewing an Image
# i = 1
# img = train_images.iloc[i].as_matrix()
# img = img.reshape((28,28))
# plt.title(train_labels.iloc[i,0])
# plt.imshow(img,cmap='gray')
# plt.show()
# Examining the Pixel Values
# plt.hist(train_images.iloc[i])
# plt.show()

# Training our model
# clf = svm.SVC()
# clf.fit(train_images,train_labels.values.ravel())
# accuracy = clf.score(test_images,test_labels.values.ravel())
# print(accuracy)

# 任何像素值均为1，否则为0
# i=1
train_images[train_images>0] = 1
test_images[test_images>0] = 1
# img = train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap="binary")
# plt.title(train_labels.iloc[i])
# plt.show()

# plt.hist(train_images.iloc[i])
# plt.show()

# Training our model
clf = KNeighborsClassifier()
clf.fit(train_images,train_labels.values.ravel())
accuracy = clf.score(test_images,test_labels.values.ravel())
print(accuracy)

test_data=pd.read_csv('E:/Kaggle_Data/Digit Recognizer/test.csv')
test_data[test_data>0] = 1

img = test_data.iloc[2].as_matrix().reshape((28,28))
plt.imshow(img,cmap="binary")
plt.title(clf.predict(test_data.iloc[2]))
plt.show()

results=clf.predict(test_data.iloc[:,:])
print(results)
df = pd.DataFrame(results)
df.index += 1
df.index.names = ['ImageId']
df.columns = ['Label']
df.to_csv('E:/Kaggle_Data/Digit Recognizer/results.csv', header=True)
