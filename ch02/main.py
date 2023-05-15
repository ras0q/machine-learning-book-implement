from perceptron import Perceptron, plot_decision_regions
import pandas as pd
import matplotlib.pyplot as plt

# rows
r1, r2, r3 = "volatile acidity", "alcohol", "quality"
# types
t1, t2 = 4, 7

s = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(s, delimiter=";")
df = df[df[r3].isin([t1, t2])]
df[r3] = df[r3].replace(t1, 0)
df[r3] = df[r3].replace(t2, 1)

t1s = df[df[r3] == 0]
t2s = df[df[r3] == 1]
plt.scatter(t1s[r1], t1s[r2], color="red", marker="o", label="0")
plt.scatter(t2s[r1], t2s[r2], color="blue", marker="x", label="1")
plt.xlabel(r1)
plt.ylabel(r2)
plt.legend(loc="upper left")
plt.savefig("ch02/ch2.png")
plt.clf()

X = df.loc[:, [r1, r2]].values
y = df.loc[:, r3].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.savefig("ch02/ch2_errors.png")
plt.clf()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel(r1)
plt.ylabel(r2)
plt.legend(loc="upper left")
plt.savefig("ch02/ch2_decision_regions.png")
plt.clf()
