import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.colors import rgb_to_hsv
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


DATA_SIZE = 4323

label_dict = {
    'daisy' : 0,
    'dandelion' : 1,
    'rose' : 2,
    'sunflower' : 3,
    'tulip' : 4
}

flower_dict = {
    0 : 'daisy',
    1 : 'dandelion',
    2 : 'rose',
    3 : 'sunflower',
    4 : 'tulip'
}


def create_histogram_vector(img, bins):
    """This function creates the histogram vector for
    every colour channel of th input image"""
    
    red_hist = np.histogram(img[:,:,0], bins=bins)
    green_hist = np.histogram(img[:,:,1], bins=bins)
    blue_hist = np.histogram(img[:,:,2], bins=bins)
    
    return np.concatenate([red_hist[0], green_hist[0], blue_hist[0]]) / (img.shape[0] * img.shape[1])
    

def create_hsv_histogram(img, bins):
    """This function creates the histogram vector for
    every colour channel of th input image"""
    
    # convert image to hsv
    img = rgb_to_hsv(img)

    red_hist = np.histogram(img[:,:,0], bins=bins)
    green_hist = np.histogram(img[:,:,1], bins=bins)
    blue_hist = np.histogram(img[:,:,2], bins=bins)
    
    return np.concatenate([red_hist[0], green_hist[0], blue_hist[0]]) / (img.shape[0] * img.shape[1])

    
def display_colour_histogram(hist, label):
    
    blue_end = len(hist)
    red_end = blue_end // 3
    green_end = red_end * 2 

    plt.bar(list(range(red_end)), hist[:red_end], color='red')
    plt.bar(list(range(red_end, green_end)), hist[red_end:green_end], color='green')
    plt.bar(list(range(green_end, blue_end)), hist[green_end:blue_end], color='blue')
    plt.title(flower_dict[label])
    plt.show()


def display_pca(hist, label, dim):
    
	pca = PCA(n_components=dim)
	X = pca.fit_transform(hist)

	colour_dict = {
	    0 : 'yellow',
	    1 : 'gray',
	    2 : 'red',
	    3 : 'black',
	    4 : 'blue'
	}

	colour_sequence = [colour_dict[x] for x in label]

	if dim == 2:
		plt.scatter(X[:, 0], X[:, 1], c=colour_sequence)
		plt.title('PCA 2 dimensions')
		plt.show()
	else:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colour_sequence)
		plt.title('PCA 3 dimensions')
		plt.show()



def extract_image_histograms(bins=32):
    """This function loads the images and computes their
    colour histograms that will be used later"""
    
    read_dir = 'flowers'
    image_histograms = np.empty((DATA_SIZE, 3 * bins), dtype=np.float32)
    image_labels = np.empty((DATA_SIZE,), dtype=np.int32)
    
    index = 0
    
    # iterate over classes
    for label, clasz in enumerate(os.listdir(read_dir)):
        
        # iterate over images of a specific class
        for img_name in os.listdir(os.path.join(read_dir, clasz)):
            
            # display once every 100 images             
            if index % 100 == 0:
                print('Processing image {0:4d}'.format(index))
            
            # load the image
            img = mpimg.imread(os.path.join(read_dir, clasz, img_name))
            
            # create colour histogram             
            # img_vector = create_histogram_vector(img, bins)
            img_vector = create_hsv_histogram(img, bins)
            image_histograms[index] = img_vector
            image_labels[index] = label_dict[clasz]
            
            index += 1
            
    return (image_histograms, image_labels)
        

def confusion_matrix(labels, predictions):
	
	# build the confussion matrix (populate the values)
	conf_matrix = np.zeros((len(flower_dict), len(flower_dict)), dtype=np.int32)
	for i in range(len(labels)):
		conf_matrix[labels[i]][predictions[i]] += 1


	# print the header of the table
	print('{0:>9} | {1:>9}'.format('', flower_dict[0]), end='')
	for i in range(1, len(flower_dict)):
		print(' | {0:>9}'.format(flower_dict[i]), end='')
	print('')

	# print the values of the table
	for i in range(len(flower_dict)):
		print('{0:>9}'.format(flower_dict[i]), end="")
		for j in range(len(flower_dict)):
			print(' | {0:>9}'.format(conf_matrix[i, j]), end="")
		print('') 
