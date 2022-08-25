import urllib.request
import pandas as pd

df = pd.read_csv("test.csv", usecols=["Avatar URL"])
i = 0
for i in range(len(df)):
    urllib.request.urlretrieve(df.loc[i, "Avatar URL"], "img_" + str(i) + ".jpg")
