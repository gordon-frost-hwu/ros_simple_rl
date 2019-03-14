import matplotlib.pyplot as plt
import numpy as np
from utilities.gaussian import gaussian1D
from srl.approximators.ann_approximator_from_scratch import ANNApproximator
plt.ion()

NUM_SAMPLES = 100
fig, ax = plt.subplots()

ax.clear()
def sigmoid(x):
    return 1.57* np.exp(-x/50)
desired_response = [(-1.0 + (1.0 - sigmoid(x))) for x in np.linspace(0, 500,NUM_SAMPLES)]
noisy_desired_response = [(-1.0 + (1.0 - sigmoid(x)) + np.random.normal((-1.0 + (1.0 - sigmoid(x))), 0.5)) for x in np.linspace(0, 500,NUM_SAMPLES)]
# noisy_desired_response = np.linspace(-3.14, 3.14,1000)

idx = 0
rewards = []
for point in desired_response:
    rewards.append(-10 + gaussian1D(noisy_desired_response[idx], point, 10.0, 2.0))
    idx += 1
def expo(x):
    return np.exp(-x/50)
ax.plot(np.linspace(0, 500, NUM_SAMPLES), desired_response, "b")
ax.plot(np.linspace(0, 500, NUM_SAMPLES), noisy_desired_response, "g")
ax.plot(np.linspace(0, 500, NUM_SAMPLES), rewards, "r--")
plt.draw()

useless_var = raw_input("press any key to continue ...")

ax.clear()
NUM_SAMPLES = 100
approx = ANNApproximator(1, 32, "tanh")
for episode in range(0, 33, 1):
    approx.setParams(list(np.load("policy_params{0}.npy".format(episode))))
    curve = []
    for i in np.linspace(-3.14, 3.14, NUM_SAMPLES):
        curve.append(approx.computeOutput(np.array(i)))
    print("Plotting episode: {0}".format(episode))
    plt.plot(np.linspace(-3.14, 3.14, NUM_SAMPLES), curve, "r-")
    plt.draw()