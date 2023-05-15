from perceptron import Perceptron, plot_decision_regions
import pandas as pd
import matplotlib.pyplot as plt

"""
Attribute Information:
  1. Id number: 1 to 214
  2. RI: refractive index
  3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-r3)
  4. Mg: Magnesium
  5. Al: Aluminum
  6. Si: Silicon
  7. K: Potassium
  8. Ca: Calcium
  9. Ba: Barium
  r3. Fe: Iron
  11. Type of glass: (class attribute)
      -- 1 building_windows_float_processed
      -- 2 building_windows_non_float_processed
      -- 3 vehicle_windows_float_processed
      -- 4 vehicle_windows_non_float_processed (none in this database)
      -- 5 containers
      -- 6 tableware
      -- 7 headlamps
"""

# rows
r1, r2, r3 = 2, 8, 10
# types
t1, t2 = 2, 7

s = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
df = pd.read_csv(s, header=None)
# select only building_windows_float_processed and vehicle_windows_float_processed
df = df[(df[r3] == t1) | (df[r3] == t2)]
print(df)
# replace 1 with 0 and 3 with 1
df[r3] = df[r3].replace(t1, 0)
df[r3] = df[r3].replace(t2, 1)

t1s = df[df[r3] == 0]
t2s = df[df[r3] == 1]
plt.scatter(t1s[r1], t1s[r2], color="red", marker="o", label=t1)
plt.scatter(t2s[r1], t2s[r2], color="blue", marker="x", label=t2)
plt.xlabel("Sodium [%]")
plt.ylabel("Magnesium [%]")
plt.legend(loc="upper left")
plt.savefig("ch02/ch2.png")
plt.clf()

X = df.iloc[:, [r1, r2]].values
y = df.iloc[:, r3].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.savefig("ch02/ch2_errors.png")
plt.clf()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Sodium [%]")
plt.ylabel("Magnesium [%]")
plt.legend(loc="upper left")
plt.savefig("ch02/ch2_decision_regions.png")
plt.clf()
