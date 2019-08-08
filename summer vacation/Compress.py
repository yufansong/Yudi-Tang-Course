from skimage import io
#import skimage as sk
from sklearn.cluster import KMeans
import numpy as np

image=io.imread(r'C:\Users\Ordinary\Desktop\tiger.png')
io.imshow(image)
io.show()

rows=image.shape[0]
cols=image.shape[1]

image=image.reshape(image.shape[0]*image.shape[1],3)
kmeans=KMeans(n_clusters=129,n_init=10,max_iter=200)
#kmeans.fit(image)

clusters=np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
labels=np.asarray(kmeans.labels_,dtype=np.uint8)
labels=labels.reshape(rows,cols)

print(clusters.shape)
np.save(r'C:\Users\Ordinary\Desktop\codebook',clusters)
io.imsave(r'C:\Users\Ordinary\Desktop\compressed_tiger.png',labels)


image=io.imread(r'C:\Users\Ordinary\Desktop\compressed_tiger.png')
io.imshow(image)
io.show()