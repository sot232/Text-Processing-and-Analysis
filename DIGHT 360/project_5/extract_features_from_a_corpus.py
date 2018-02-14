from glob import glob
import re
register_categories = []
files = []
wordCount = []
for filename in glob('Mini-CORE/*.txt'):
    with open(filename) as f:
        register_categories.append(filename[12:14])
        clear = re.compile('<.*?>')
        clearText = re.sub(clear, '', f.read())
        files.append(clearText)
        wordCount.append(len(clearText))

pronouns = ['i', 'you', 'he', 'she', 'it', 'we',
            'they', 'what', 'who', 'me', 'him',
            'her', 'us', 'them', 'whom', 'mine',
            'yours', 'his', 'hers', 'ours', 'theirs'
            'this', 'that', 'these', 'those',
            'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'themselves',
            'each other', 'one another',
            'another', 'everybody', 'anything',
            'none', 'anybody', 'everyone', 'everything',
            'nobody', 'nothing', 'none', 'other',
            'others', 'somebody', 'someone', 'something',
            '1st', '2nd', '3rd', 'first', 'second', 'third']
punctuations = ['.', ',', '?', '!', "'", '"', ':', ';',
                '-', '_']

pronoun_count_list = []
acronym_count_list = []
punctuation_count_list = []

for file in files:
    pronoun_count = 0
    acronym_count = 0
    punctuation_count = 0
    for w in file:
        if w.lower() in pronouns:
            pronoun_count += 1
        if w.isupper():
            acronym_count += 1
        if w in punctuations:
            punctuation_count += 1
    pronoun_count_list.append(pronoun_count)
    acronym_count_list.append(acronym_count)
    punctuation_count_list.append(punctuation_count)

i = 0
print("filename\t<Pronoun>\t<Acronym>\t<Punctuation>\tregister")
for category in register_categories:
    if not(pronoun_count_list[i] == 0):
        ratePronoun = pronoun_count / pronoun_count_list[i]
    if not(acronym_count_list[i] == 0):
        rateAcronym = acronym_count / acronym_count_list[i]
    if not(punctuation_count_list[i] == 0):
        ratePunctuation = punctuation_count / punctuation_count_list[i]
    print('1+' + str(category) + '+....txt\t' + str(ratePronoun) +
          '\t' + str(rateAcronym) + '\t' + str(ratePunctuation) +
          '\t' + str(category))
    i += 1
