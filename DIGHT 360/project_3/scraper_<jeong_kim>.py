"""

Choose any website(s)
that you would like to scrape
and write a script to get the html
and save it to a file
(<filename>.html).

"""

import requests as r

wiki_korea_url = 'https://en.wikipedia.org/wiki/Korea'
headers = {'user-agent': 'Jeong Kim (tjtekrkchlrh@gmail.com)'}
response = r.get(wiki_korea_url, headers=headers)

path = '/Users/jeong-ugim/Documents/BYU/\
2018 Winter Semester/DIGHT 360/DIGHT 360/\
project_3/wiki_korea.html'

file = open(path, "w", encoding='utf8')
print(response.text)

file.write(response.text)
file.close()
