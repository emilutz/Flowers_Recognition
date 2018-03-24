import os
import sys
import numpy as np
from functions import display_pca


if __name__ == "__main__":

	if len(sys.argv) != 2:
		sys.exit('Insert a number between 2 and 3')

	try:
		dim = int(sys.argv[1])
		if dim < 2 or dim > 3:
			sys.exit('Insert a number between 2 and 3')
	except ValueError:
		sys.exit('Insert a number between 2 and 3')


	# load the histograms and the labels
	histograms = np.load(os.path.join('colour_histograms','histograms.dat'))
	labels = np.load(os.path.join('colour_histograms','labels.dat'))

	# call the PCA function
	display_pca(histograms, labels, dim)
