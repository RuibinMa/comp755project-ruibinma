import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
#scores_filepath = '/home/ruibinma/Desktop/GeM-Colon-whiten.bin'
scores_filepath = '/home/ruibinma/Desktop/Vocab-Colon.bin'
scores = np.load(scores_filepath)
maxscore = np.max(scores)

for i in range(scores.shape[0]):
    plt.ylim((0., maxscore))

    interval = scores[i, :i+1]
    temp = maximum_filter(interval, size=20, mode='constant', cval=0.0)
    positions = np.where((temp == interval) & (interval > 0) & (interval > 0.4))[0]
    if positions[-1] == i:
        positions = positions[:-1]
    values = scores[i, positions]
    plt.plot(positions, values, 'ro', scores[i,:])
    print("{} -> {}".format(i, positions))
    plt.show()