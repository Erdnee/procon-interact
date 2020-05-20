import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#policy = np.loadtxt('plot_data_policy.txt', dtype = float)
policy_dynamic = np.loadtxt('plot_data_policy_dynamic.txt',dtype=float)


plt.ylabel("Total Reward")
plt.xlabel("Episode number")
plt.title("Policy gradient on dynamic environment")
plt.plot(range(policy_dynamic.size),policy_dynamic, ".", color='orange', label='non stationary environment')
#plt.plot(range(policy.size),policy,".",label = 'stationary environment')
plt.show()