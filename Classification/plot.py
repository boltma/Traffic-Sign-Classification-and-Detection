import matplotlib.pyplot as plt
import pandas as pd

a = pd.read_csv("accuracy.csv")
data = a[0:100]
plt.plot(data["epoch"], data["accuracy"])
plt.grid(True, linestyle = '-.')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.show()
    