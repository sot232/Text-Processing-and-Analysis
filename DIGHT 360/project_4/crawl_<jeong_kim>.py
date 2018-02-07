"""
Author: Jeong Kim
create files according to the url name
each file has HTML
"""


def get_rnlp(url_name):
    import requests as r
    # testing : url_name = 'aa.html'
    initial_url = 'http://reynoldsnlp.com/scrape/' + str(url_name)
    headers = {'user-agent': 'Jeong Kim (tjtekrkchlrh@gmail.com)'}
    response = r.get(initial_url, headers=headers)
    folder_path = '/Users/jeong-ugim/Documents/BYU/'\
    '2018 Winter Semester/DIGHT 360/DIGHT 360/'\
    'project_4/scrape/' + str(url_name)
    file = open(folder_path, 'w', encoding='utf8')
    print(response.text)
    file.write(response.text)
    file.close()

"""
return a list of hrefs
for example, ['aa.html', 'ba.html']
"""


def get_hrefs(url_name):
    from bs4 import BeautifulSoup
    # testing : url_name = 'aa.html'
    folder_path = '/Users/jeong-ugim/Documents/BYU/' \
    '2018 Winter Semester/DIGHT 360/DIGHT 360/' \
    'project_4/scrape/' + str(url_name)
    file = open(folder_path, 'r', encoding='utf8')
    html = file.read()
    html_soup = BeautifulSoup(html, 'html.parser')
    hrefs_lists = []
    for link in html_soup.find_all('a'):
        key_hrefs = str(link.get('href')[-7:])
        hrefs_lists.append(key_hrefs)
    return hrefs_lists

"""
check if the two lists are identical
keep track of three boolean values
1) isExit : check if we want to exit the while loop later
2) isAdd : check if we add any element to the old list
3) isHave : check if any element from the new list
            is in the old list
Return the isExit value
"""


def check_dupl_add(new_list, old_list):
    isExit = False
    isAdd = False
    # if two lists are identical, then return true
    # otherwise, return false
    for new_each in new_list:
        isHave = False
        for old_each in old_list:
            if (new_each == old_each):
                isHave = True
                exit
        if not(isHave):
            old_list.append(new_each)
            isAdd = True
    if not(isAdd):
        isExit = True
    return isExit

"""
get_rnlp : create a file that contains HTML
get_hrefs : extract a small portion from HTML and make a list
check_dupl_add : check for duplicates and add a new element to a list
a while loop :
    sleep for one or two seconds befor calling the get_rnlp function
    create a list that contains some parts of a tag,
    and check if the new list is the same as the original list,
    if so, exit the while loop;
    otherwise, add new elements to the original list,
    and run another loop
"""


def main():
    from time import sleep
    from random import randint
    # inital step
    get_rnlp('aa.html')
    list_hrefs = get_hrefs('aa.html')
    print(list_hrefs)
    isExit = False
    # extract a-tags
    # until there is no longer a new tag
    while (True):
        for each in list_hrefs:
            sleep(randint(1, 2))
            get_rnlp(each)
            new_list_hrefs = get_hrefs(each)
            if (check_dupl_add(new_list_hrefs, list_hrefs)):
                isExit = True
        if (isExit):
            break

# run the main function
if __name__ == "__main__":
    main()
