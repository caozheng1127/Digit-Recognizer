import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm


# labeled_images = pd.read_csv('E:/PycharmProjects/files/train.csv')
# images = labeled_images.iloc[0:5000,1:]
# labels = labeled_images.iloc[0:5000,:1]
# train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
labeled_images = pd.read_csv('E:/PycharmProjects/files/train.csv')
images = labeled_images.iloc[:, 1:]
labels = labeled_images.iloc[:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# i = 1
# img = train_images.iloc[i].as_matrix()
# img = img.reshape((28, 28))
# plt.imshow(img, cmap='gray')
# plt.title(train_labels.iloc[i, 0])
# # plt.show()
#
#
# plt.hist(train_images.iloc[i])
# plt.show()


# clf = svm.SVC()
# clf.fit(train_images, train_labels.values.ravel())
# result = clf.score(test_images,test_labels)
# #print('%10.20f' % result)


test_images[test_images>0]=1
train_images[train_images>0]=1

# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])
# plt.show()


# plt.hist(train_images.iloc[i])
# plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
result = clf.score(test_images,test_labels)
# print(result)


test_data=pd.read_csv('E:/PycharmProjects/files/test.csv')
test_data[test_data>0]=1
# results=clf.predict(test_data[0:5000])
results = clf.predict(test_data.iloc[:, :])
# print(results)


df = pd.DataFrame(results)
df.index.name='ImageId'
df.index += 1
df.columns=['Label']
df.to_csv('results(all).csv', header=True)


