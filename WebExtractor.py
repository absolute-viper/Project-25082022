import urllib.request
import pandas as pd


df = pd.read_csv("scrap_1.csv")
print(df)
for i in range(295, len(df)):
    urllib.request.urlretrieve(df.loc[i, "Img_URL"], ("DATA/"+df.loc[i, "Username"] + ".jpg"))
print("Download Complete")
