from ucimlrepo import fetch_ucirepo
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

# fetch dataset
iris = fetch_ucirepo(id=53)
originaldf = iris.data.original

sns.pairplot(originaldf, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()