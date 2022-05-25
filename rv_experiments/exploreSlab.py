import pickle
from matplotlib import pyplot as plt


base_dir = 'data/datasets/slab/'
total_slabs = 7
slab_noise = 0.1
spur_corr = 0.1
case = 'train'
seed = 2
data_dir = base_dir + 'total_slabs_' + str(total_slabs) + '_slab_noise_' + str(slab_noise) + '_spur_corr_' + str(
    spur_corr) + '_case_' + str(case) + '_seed_' + str(seed)

with open(data_dir + '.pickle', 'rb') as fname:
    data = pickle.load(fname)

X = data['X'].numpy()
Y = data['Y'].numpy()
plt.plot(X[Y==1][:,0], X[Y==1][:,1], 'rx')
plt.plot(X[Y==0][:,0], X[Y==0][:,1], 'gx')
plt.show()
