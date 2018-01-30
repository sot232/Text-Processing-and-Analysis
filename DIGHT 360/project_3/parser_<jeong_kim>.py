import re

path = '/Users/jeong-ugim/Documents/BYU/2018 Winter Semester/DIGHT 360/project_3/wiki_korea.html'
path_w = '/Users/jeong-ugim/Documents/BYU/2018 Winter Semester/DIGHT 360/project_3/wiki_korea_parsed.txt'
wiki_korea_file = open(path, "r",encoding='utf8')
wiki_korea = wiki_korea_file.read()

testRE = 'id="Music".*?</p>'
test_match = re.search(testRE, wiki_korea, re.S)

new_file = open(path_w,"w",encoding='utf8')
new_file.write(test_match.group(0))

new_file.close()
wiki_korea_file.close()