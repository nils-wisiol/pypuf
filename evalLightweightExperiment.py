import numpy as np
import matplotlib.pyplot as plt
import matplotlib




#result =  np.load('experimentResults_stages_64_xors_4_trainSetSize_40000_time_2017_09_21__15_28_22')
result =  np.load('experimentResults_stages_64_xors_5_trainSetSize_200000_time_2017_10_04__16_48_08')

numTrials = result[:, 4]


f = plt.figure()
#plt.hist(result[:, 4], 20)
plt.hist(result[:, 4], np.arange(-0.5, 20.5, 1), edgecolor='black', linewidth=1.2)

plt.xlabel('Number Required Tries')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 20, 2))
plt.axes().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
plt.show()
f.savefig('64_5_sortedListAccRank.pdf', bbox_inches='tight')
