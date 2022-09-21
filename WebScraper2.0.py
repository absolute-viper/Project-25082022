import csv

import instaloader
from instaloader import Profile

L = instaloader.Instaloader()
USER = 'ritik._.ks'
PASSWORD = 'pass'
L.login(USER, PASSWORD)

USERNAME = 'ig_rajpatel'
profile = Profile.from_username(L.context, USERNAME)

f = open('scrap_1.csv', 'w', encoding='UTF8')
writer = csv.writer(f)
header = ['Username', 'Img_URL']
writer.writerow(header)
print("{} follows these profiles:".format(profile.username))
i = 0
for follower in profile.get_followers():
    row = [follower.username, follower.get_profile_pic_url()]
    print(i)
    i += 1
    writer.writerow(row)
f.close()
print("Process Completed")
