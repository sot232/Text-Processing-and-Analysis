# Find Nominalized Verbs
# Jeongwoo Kim

import re

# passages

passage = "The inequity in the distribution \
of wealth in Australia \
            is yet another indicator of Australia's \
            lack of egalitarianism. \
            In 1985, 20% of the Australian population \
            owned 72.2% of the wealth with \
            the top 50% owning 92.1% (Raskall, 1988: \
            287: ). Such a significant skew \
            in the distribution of wealth indicates \
            that, at least in terms of economics, \
            there is an established class system in \
            Australia. McGregor (1988) argues that \
            Australian society can be categorised into \
            three levels: the Upper; Middle and \
            Working classes. In addition, it has been shown \
            that most Australians continue to \
            remain in the class into which they were born \
            (McGregor,1988: 156) despite \
            arguments about the ease of social mobility in \
            Australian society (Fitzpatrick, \
            1994). The issue of class and its inherent inequity, \
            however, is further compounded \
            by factors such as race and gender within and across \
            these class divisions. \
            During a historic live broadcast Tuesday, January 16, \
            from the annex of the Salt \
            Lake Temple to all members of the Church, newly ordained \
            prophet President Russell \
            M. Nelson reassured each member of the Church that \
            “whatever your concerns, whatever \
            your challenges, there is a place for you in this, \
            the Lord’s Church. President Nelson \
            said he and his counselors in the First Presidency, \
            President Dallin H. Oaks and \
            President Henry B. Eyring, chose to broadcast \
            from the temple to emphasize their \
            wish for members to be endowed with power in a \
            house of the Lord, sealed as \
            families—faithful to covenants made in a temple,” \
            which are key to strengthening \
            your life, your marriage and family, and your ability \
            to resist the attacks \
            of the adversary. Your worship in the temple and \
            your service there for \
            your ancestors will bless you with increased \
            personal revelation and peace \
            and will fortify your commitment to stay on \
            the covenant path, he said. \
            Urging members to keep on the covenant path, \
            President Nelson said, Your \
            commitment to follow the Savior by making \
            covenants with Him and then keeping \
            those covenants will open the door to every \
            spiritual blessing and privilege \
            available to men, women, and children everywhere, \
            he said. Expressing thanks \
            for parents who are serious about their \
            commitment to righteous, intentional \
            parenting, President Nelson shared how a \
            4-year-old boy named Benson prayed that \
            as the prophet he would “be brave and not \
            scared that he’s new, and help him to \
            grow up to be healthy and strong. He said \
            he was thankful for every parent, teacher, \
            and member who carries heavy burdens \
            and yet serves so willingly."

# zero-change word lists
zero_change = {'murder', 'use', 'change'}

"""
gerunds_form: It includes something like willingly.
agent_form: It includes something like categorised.
recipient_form: It includes a word like three.
other_form: it didn't include $ because tion, sion, ment, ence, ance forms
            almost always comes at the end of word.
            Thus leavning the regex without $ doesn't seem to be riskly.
no_chance_form: This form mostly filters out what we want.
zero_change: As a computer, it is difficult to
             discern whether a word is used as a verb or noun.
             So I just give words to a computer
             and let it find the words in passages.
"""


def main():
    count = 0

    # asking if the word ends with ing and
    # there are at least two alphabetical characters
    gerunds_re = r'\w{2,}ing[^A-Za-z]'
    # asking if the word ends with 'or' or 'ors'
    # and doesn't start with uppercase
    agent_re = r'[^A-Z][a-z]{3,}or|ors[^A-Za-z]'
    # asking if the word ends with ee
    # and there are at least three alphabetical characters
    recipient_re = r'[a-z]{3,}ee[^A-Za-z]'
    # asking if the word includes
    # tion, sion, ment, ence, or ance
    other_re = r'\w{3,}tion | \w{3,}sion \
               | \w{3,}ment | \w{3,}ence | \w{3,}ance'
    # asking if the word is exactly matching
    zero1_re = r'(?:^|\W)murder(?:$|\W)'
    zero2_re = r'(?:^|\W)use(?:$|\W)'
    zero3_re = r'(?:^|\W)change(?:$|\W)'

    gerunds_result = re.findall(gerunds_re, passage)
    agent_result = re.findall(agent_re, passage)
    recipient_result = re.findall(recipient_re, passage)
    other_result = re.findall(other_re, passage)
    zero1_result = re.findall(zero1_re, passage)
    zero2_result = re.findall(zero2_re, passage)
    zero3_result = re.findall(zero3_re, passage)

    print('Gerunds_result: ')
    for i in gerunds_result:
        print('\t', i)

    print('Agent_result: ')
    for i in agent_result:
        print('\t', i)

    print('Recipient_result: ')
    for i in recipient_result:
        print('\t', i)

    print('Other_result: ')
    for i in other_result:
        print('\t', i)

    if (zero1_result):
        count += len(zero1_result)
        print(zero1_result)
    if (zero2_result):
        count += len(zero2_result)
        print(zero2_result)
    if (zero3_result):
        count += len(zero3_result)
        print(zero3_result)

    count += len(gerunds_result)
    count += len(agent_result)
    count += len(recipient_result)
    count += len(other_result)

    print('Total number of nominalization : ', count)

if __name__ == '__main__':
    main()
