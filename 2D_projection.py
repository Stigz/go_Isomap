import csv
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.offline as py
import numpy as np

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

results = []
with open("result2.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

x = results[0]
y = results[1]

fig, ax = plt.subplots()
ax.scatter(x, y)

plot_url = py.plot_mpl(fig, filename="mpl-scatter")