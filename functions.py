import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


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
    
    
def display_colour_histogram(hist, label):
    
    blue_end = len(hist)
    red_end = blue_end // 3
    green_end = red_end * 2 

    plt.bar(list(range(red_end)), hist[:red_end], color='red')
    plt.bar(list(range(red_end, green_end)), hist[red_end:green_end], color='green')
    plt.bar(list(range(green_end, blue_end)), hist[green_end:blue_end], color='blue')
    plt.title(flower_dict[label])
    plt.show()


def display_pca(hist, label):
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(hist)

    colour_dict = {
        0 : 'yellow',
        1 : 'gray',
        2 : 'red',
        3 : 'black',
        4 : 'blue'
    }

    colour_sequence = [colour_dict[x] for x in label]

    plt.scatter(X[:, 0], X[:, 1], c=colour_sequence)
    plt.show()


def extract_image_histograms(bins=100):
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
            img_vector = create_histogram_vector(img, bins)
            image_histograms[index] = img_vector
            image_labels[index] = label_dict[clasz]
            
            index += 1
            
    return (image_histograms, image_labels)
        