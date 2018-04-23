import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


IMG_SIZE = 150
CHANNELS = 3
DATA_SIZE = 4323

label_dict = {
    'daisy' : 0,
    'dandelion' : 1,
    'rose' : 2,
    'sunflower' : 3,
    'tulip' : 4
}

reading_path = 'flowers_cropped'
writing_path = 'tensorflow_data'


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf(data, labels):

	writing = os.path.join(writing_path, 'training.tfrecords')
	writer = tf.python_io.TFRecordWriter(writing)

	for index in range(len(labels)):
		data_sample = data[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
		    'label': _int64_feature(int(labels[index])),
		    'image': _bytes_feature(data_sample)}))
		writer.write(example.SerializeToString())
		
	writer.close()


def read_data():

	images_data = np.empty((DATA_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
	images_labels = np.empty((DATA_SIZE,), dtype=np.int32)
	index = 0

	for flower_class in os.listdir(reading_path):
		for img_name in os.listdir(os.path.join(reading_path, flower_class)):
			
			# read image as bgr
			image_bgr = cv2.imread(os.path.join(reading_path, flower_class, img_name),
								   cv2.IMREAD_COLOR)

			# convert to rgb
			image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

			images_data[index] = image
			images_labels[index] = label_dict[flower_class]
			
			index += 1
			print("Sample {0:4d}".format(index))

	return (images_data, images_labels)


def shuffle_data(data, labels):

	# find a random permutation
	permutations = np.random.permutation(len(labels))
	data = data[permutations]
	labels = labels[permutations]

	return (data, labels)


def split_data(data, labels, trn, val):

	train_point = int(trn * len(labels))
	val_point = int((val + trn) * len(labels))

	training_data = data[:train_point]
	training_labels = labels[:train_point]

	validation_data = data[train_point:val_point]
	validation_labels = labels[train_point:val_point]

	testing_data = data[val_point:]
	testing_labels = labels[val_point:]

	return (training_data, training_labels,
		    validation_data, validation_labels,
		    testing_data, testing_labels)


if __name__ == '__main__':
    
	data, labels = read_data()
	data, labels = shuffle_data(data, labels)

	tr_D, tr_L, vl_D, vl_L, ts_D, ts_L = split_data(data, labels, 0.75, 0.15)

	# convert training data to tfrecords
	convert_to_tf(tr_D, tr_L)

	# serialize validation and test
	vl_D.dump(os.path.join(writing_path, "validation_data.dat"))
	vl_L.dump(os.path.join(writing_path, "validation_labels.dat"))
	ts_D.dump(os.path.join(writing_path, "testing_data.dat"))
	ts_L.dump(os.path.join(writing_path, "testing_labels.dat"))
