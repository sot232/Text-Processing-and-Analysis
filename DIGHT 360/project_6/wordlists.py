from glob import glob
import re
from nltk import FreqDist

function_words = """AGAIN AGO ALMOST ALREADY ALSO ALWAYS ANYWHERE
                    BACK ELSE EVEN EVER EVERYWHERE FAR HENCE HERE
                    HITHER HOW HOWEVER JUST NEAR NEARBY NEARLY NEVER
                    NOT NOW NOWHERE OFTEN ONLY QUITE RATHER SOMETIMES
                    SOMEWHERE SOON STILL THEN THENCE THERE THEREFORE
                    THITHER THUS TODAY TOMORROW TOO UNDERNEATH VERY
                    WHEN WHENCE WHERE WHITHER WHY YES YESTERDAY YET
                    AM ARE AREN'T BE BEEN BEING CAN CAN'T COULD COULDN'T
                    DID DIDN'T DO DOES DOESN'T DOING DONE DON'T GET GETS
                    GETTING GOT HAD HADN'T HAS HASN'T HAVE HAVEN'T
                    HAVING HE'D HE'LL HE'S I'D I'LL I'M IS I'VE ISN'T
                    IT'S MAY MIGHT MUST MUSTN'T OUGHT OUGHTN'T SHALL
                    SHAN'T SHE'D SHE'LL SHE'S SHOULD SHOULDN'T THAT'S
                    THEY'D THEY'LL THEY'RE WAS WASN'T WE'D WE'LL WERE
                    WE'RE WEREN'T WE'VE WILL WON'T WOULD WOULDN'T
                    YOU'D YOU'LL YOU'RE YOU'VE
                    ABOUT ABOVE AFTER ALONG ALTHOUGH AMONG
                    AND AROUND AS AT BEFORE BELOW BENEATH
                    BESIDE BETWEEN BEYOND BUT
                    BY DOWN DURING EXCEPT FOR FROM IF IN
                    INTO NEAR NOR OF OFF ON OR OUT OVER
                    ROUND SINCE SO THAN THAT
                    THOUGH THROUGH TILL TO TOWARDS UNDER
                    UNLESS UNTIL UP WHEREAS WHILE WITH WITHIN WITHOUT
                    A ALL AN ANOTHER ANY ANYBODY ANYTHING BOTH EACH
                    EITHER ENOUGH EVERY EVERYBODY EVERYONE EVERYTHING
                    FEW FEWER HE HER HERS HERSELF HIM HIMSELF HIS
                    I IT ITS ITSELF LESS MANY ME MINE MORE MOST
                    MUCH MY MYSELF NEITHER NO NOBODY NONE NOONE
                    NOTHING OTHER OTHERS OUR OURS OURSELVES SHE SOME
                    SOMEBODY SOMEONE SOMETHING SUCH THAT THE THEIR
                    THEIRS THEM THEMSELVES THESE THEY THIS THOSE US
                    WE WHAT WHICH WHO WHOM WHOSE YOU
                    YOUR YOURS YOURSELF YOURSELVES
                    BILLION BILLIONTH EIGHT EIGHTEEN EIGHTEENTH EIGHTH
                    EIGHTIETH EIGHTY ELEVEN ELEVENTH FIFTEEN FIFTEENTH
                    FIFTH FIFTIETH FIFTY FIRST FIVE FORTIETH FORTY FOUR
                    FOURTEEN FOURTEENTH FOURTH HUNDRED HUNDREDTH LAST
                    MILLION MILLIONTH NEXT NINE NINETEEN NINETEENTH
                    NINETIETH NINETY NINTH ONCE ONE SECOND SEVEN
                    SEVENTEEN SEVENTEENTH SEVENTH SEVENTIETH SEVENTY
                    SIX SIXTEEN SIXTEENTH SIXTH SIXTIETH SIXTY TEN
                    TENTH THIRD THIRTEEN THIRTEENTH THIRTIETH THIRTY
                    THOUSAND THOUSANDTH THREE THRICE TWELFTH
                    TWELVE TWENTIETH TWENTY TWICE TWO
                    A ABOUT ABOVE AFTER AGAIN AGO ALL ALMOST ALONG
                    ALREADY ALSO ALTHOUGH ALWAYS AM AMONG AN AND
                    ANOTHER ANY ANYBODY ANYTHING ANYWHERE ARE AREN'T
                    AROUND AS AT BACK ELSE BE BEEN BEFORE BEING BELOW
                    BENEATH BESIDE BETWEEN BEYOND BILLION BILLIONTH
                    BOTH EACH BUT BY CAN CAN'T COULD COULDN'T DID
                    DIDN'T DO DOES DOESN'T DOING DONE DON'T DOWN
                    DURING EIGHT EIGHTEEN EIGHTEENTH EIGHTH EIGHTIETH
                    EIGHTY EITHER ELEVEN ELEVENTH ENOUGH EVEN EVER
                    EVERY EVERYBODY EVERYONE EVERYTHING EVERYWHERE
                    EXCEPT FAR FEW FEWER FIFTEEN FIFTEENTH FIFTH
                    FIFTIETH FIFTY FIRST FIVE FOR FORTIETH FORTY
                    FOUR FOURTEEN FOURTEENTH FOURTH HUNDRED FROM
                    GET GETS GETTING GOT HAD HADN'T HAS HASN'T
                    HAVE HAVEN'T HAVING HE HE'D HE'LL HENCE HER
                    HERE HERS HERSELF HE'S HIM HIMSELF HIS HITHER
                    HOW HOWEVER NEAR HUNDREDTH I I'D IF I'LL I'M
                    IN INTO IS I'VE ISN'T IT ITS IT'S ITSELF JUST
                    LAST LESS MANY ME MAY MIGHT MILLION MILLIONTH
                    MINE MORE MOST MUCH MUST MUSTN'T MY MYSELF NEAR
                    NEARBY NEARLY NEITHER NEVER NEXT NINE NINETEEN
                    NINETEENTH NINETIETH NINETY NINTH NO NOBODY
                    NONE NOONE NOTHING NOR NOT NOW NOWHERE OF
                    OFF OFTEN ON OR ONCE ONE ONLY OTHER OTHERS
                    OUGHT OUGHTN'T OUR OURS OURSELVES OUT OVER
                    QUITE RATHER ROUND SECOND SEVEN SEVENTEEN
                    SEVENTEENTH SEVENTH SEVENTIETH SEVENTY
                    SHALL SHAN'T SHE'D SHE SHE'LL SHE'S SHOULD
                    SHOULDN'T SINCE SIX SIXTEEN SIXTEENTH SIXTH
                    SIXTIETH SIXTY SO SOME SOMEBODY SOMEONE
                    SOMETHING SOMETIMES SOMEWHERE SOON STILL
                    SUCH TEN TENTH THAN THAT THAT THAT'S THE
                    THEIR THEIRS THEM THEMSELVES THESE THEN
                    THENCE THERE THEREFORE THEY THEY'D THEY'LL
                    THEY'RE THIRD THIRTEEN THIRTEENTH THIRTIETH
                    THIRTY THIS THITHER THOSE THOUGH THOUSAND
                    THOUSANDTH THREE THRICE THROUGH THUS TILL
                    TO TOWARDS TODAY TOMORROW TOO TWELFTH TWELVE
                    TWENTIETH TWENTY TWICE TWO UNDER UNDERNEATH
                    UNLESS UNTIL UP US VERY WHEN WAS WASN'T WE
                    WE'D WE'LL WERE WE'RE WEREN'T WE'VE WHAT WHENCE
                    WHERE WHEREAS WHICH WHILE WHITHER WHO WHOM
                    WHOSE WHY WILL WITH WITHIN WITHOUT WON'T
                    WOULD WOULDN'T YES YESTERDAY YET YOU YOUR
                    YOU'D YOU'LL YOU'RE YOURS YOURSELF
                    YOURSELVES YOU'VE"""
set_function_words = list(set(function_words.lower().split(" ")))
set_function_words += " "

files = []
i = 1
path = 'output/'

for filename in glob('Mini-CORE/*.txt'):
    with open(filename, 'r', encoding='utf8') as f:
        w_file = open(path + str(i) + ".md", 'w', encoding='utf8')
        clear = re.compile('<.*?>')
        n_clear = re.compile('\n')
        preClearText = re.sub(clear, '', f.read())
        clearText = re.sub(n_clear, '', preClearText).lower()
        # w_file.write(clearText + '\n\n')
        tokens = sorted(list(clearText.split(" ")))
        fd = FreqDist()
        for word in tokens:
            if word not in set_function_words:
                fd.update([word])
        fd_sorted = sorted(fd, key=fd.get, reverse=True)
        w_file.write('Sorted by values :\n')
        for word in fd_sorted:
            w_file.write(str(word) + ' ')
        w_file.write('\n\nList :\n')
        for word in fd:
            w_file.write(str(word) + ' : ' + str(fd[word]) + '\n')
        w_file.close()
        i += 1
