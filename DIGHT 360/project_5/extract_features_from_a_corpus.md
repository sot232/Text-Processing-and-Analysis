In Learning Suite under Content I have uploaded a corpus called Mini-CORE (Corpus of Online Registers of English). It contains 1,200 texts, 200 from each of the following register categories:

IN: Informational (Wikipedia)

IP: Informational Persuasion

LY: Song Lyrics

NA: News reports

OP: Opinion blogs

SP: Interview transcripts

The filenames are long. You can ignore everything except the first two letter code (after the 1+). That will tell you the register. For example, 1+IP+DS+IP-IP-IP-IP+DS-DS-DS-DS+NNNY+0583255.txt is an IP text.

Each file has a header with human annotator ratings and other information. You should only include lines that begin with <h> or <p> in your analysis (but remove the <h> or the <p>).

For this assignment, you will:

Select 3 linguistic features that you think will vary across register categories.
Identify those features (make sure to check accuracy)
Calculate normed rates of occurrence (per 1,000 words)
Print them to a tab-separated file along with a column for 'Register' which should contain a single two letter code for the register category (i.e. extract the two letter register code from the filename). Your output should be 1601 lines long and should look something like:
filename	<feat1>	<feat2>	<feat3>	register
1+IN+....txt	15.2	131.3	22.4	IN
1+HI+....txt	14.6	119.6	26.9	HI
Change <feat1>, <feat2>, and <feat3> to actual labels of your selected features. Here are some features you can consider counting:

pronouns (all pronouns or 1st, 2nd, or 3rd)
proper nouns
past tense
modals
contractions
punctuation (e.g., ? or !)
 

Turn in a brief writeup, along with a link to your script and your output file. (Unless your script and output are on github, I prefer that you attach the files on Learning Suite for this assignment.)