Write a brief summary explaining how you did the following task, and include a link to your code. State what pattern you see in the files you saved. Are any files missing, or is the pattern complete?

Write a function called `get_rnlp` that requests an arbitrary filename (filename is an argument of the function) from `http://reynoldsnlp.com/scrape/` and saves the file in the local folder `scrape/`. (If you give the argument "aa.html", it should get the HTML from "http://reynoldsnlp.com/scrape/aa.html".)

Write a function called `get_hrefs` that opens a local html file (filename is an argument of the function), parses it using `BeautifulSoup` and returns a list of href values for all <a> tags in the file. (If you give the argument "aa.html", it should get the HTML from the local file "scrape/aa.html" and return a list of hrefs.)

The two functions above will be used for the following. Write a loop that will iteratively request a url, parse it, extract the hrefs, and then do the same thing with all of those hrefs. Begin with 'aa.html'.

There are many duplicates on the website. In your loop, be sure that you check whether you have already saved a page before you request it.

IMPORTANT! Follow all ethical standards for scraping, especially the ~2-sec pause between requests.