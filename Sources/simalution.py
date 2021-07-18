import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class ISimulation:
    def normalizedRewards(self, **kwargs):
        sequences = kwargs["GameSequence"]

        sinjuries = np.array([ x._values.injuries for x in sequences ], np.float)
        scost = [ np.array([ sinjuries[x][kwargs["PlayerID"]] + sinjuries[x + 1][0],
                    sinjuries[x][kwargs["EnemyID"]] + sinjuries[x + 1][1]
                    ], np.float)
                  for x in range(len(sinjuries) - 1)
                  ]

        plt.plot(range(len(scost)), gaussian_filter1d([ x[0] for x in scost ], 1), label="Cost Injuries Player")
        plt.plot(range(len(scost)), gaussian_filter1d([ x[1] for x in scost ], 1), label="Cost Injuries Enemy")
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('Cost')
        plt.legend()
        plt.savefig('./Schema/scost.png')

