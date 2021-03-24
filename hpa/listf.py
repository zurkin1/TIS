import numpy as np
import os


np.set_printoptions(threshold=np.inf)
for filename in os.listdir('hpa_cell_mask'):
	data = np.load('hpa_cell_mask/'+filename)
	print(data['arr_0'].shape)
	break