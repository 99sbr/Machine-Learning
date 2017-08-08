import random
import matplotlib.pyplot as plt

x = []
y1 = []
y2 = []

for i in range(1000):
	winner_prob = random.uniform(0, 1)
	# Additional bias to the winner
	if winner_prob < 0.5:
		winner_prob += 0.5
	elif winner_prob < 0.8:
		winner_prob += 0.2
	runner_prob = 1 - winner_prob
	x.append(i)
	y1.append(winner_prob)
	y2.append(runner_prob)

fig = plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
fig = plt.gcf()
fig.savefig("winner_vs_runner.png")
fig.clf()
