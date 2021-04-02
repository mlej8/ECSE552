import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# set axis to integer
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

mlp = [0.3115,0.3570,0.3890,0.4164,0.4411]
lstm = [0.3076,0.3390,0.3692,0.3906,0.4037]
cnn = [0.3164,0.3527,0.3839,0.4105, 0.4375]

axis = [i for i in range(len(cnn))] 
plt.plot(axis, cnn, label='cnn')
plt.plot(axis, lstm, label='lstm')
plt.plot(axis, mlp, label='mlp')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('RMSE Loss')
plt.title("Average RMSE over number of time steps for k = 4")
plt.savefig("./timestep.png")
plt.show()