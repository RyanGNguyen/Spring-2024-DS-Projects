import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue, PriorityQueue
from urllib import parse, request
from Levenshtein import ratio


logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')

def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub(r'\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


def parse_links_sorted(root, html):
    pq = PriorityQueue()
    for link, title in parse_links(root, html): 
        if not is_self_referencing(link, root):
            score = -ratio(link, root) 
            pq.put((score, (link, title))) 
    while not pq.empty():
        yield pq.get()[1]


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))

def get_sorted_links(url):
    res = request.urlopen(url)
    return list(parse_links_sorted(url, res.read()))

def is_self_referencing(cur, ref):
    parsed_cur = parse.urlparse(cur)
    parsed_ref = parse.urlparse(ref)
    if re.search(parsed_ref.path.split('/')[-1], parsed_cur.path):
        return True
    return False

def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    links = get_links(url)
    url_netloc = parse.urlparse(url).netloc
    filtered = []
    for link, title in links:
        if parse.urlparse(link).netloc != url_netloc and not is_self_referencing(link, url):
            filtered.append((link, title))
    return filtered


def crawl(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    queue = Queue()
    queue.put(root)

    visited = set()
    extracted = []
    counter = 0

    while not queue.empty() and counter < 100:
        url = queue.get()
        try:
            req = request.urlopen(url)
            html = req.read()
            
            if wanted_content and not any(re.search(c, req.headers['Content-Type']) for c in wanted_content):
                continue
            
            visited.add(url)
            visitlog.debug(url)
            
            if re.search(r'html|plain', req.headers['Content-Type']):
                for ex in extract_information(url, html):
                    extracted.append(ex)
                    extractlog.debug(ex)

            for link, title in get_sorted_links(url):
                if within_domain and parse.urlparse(link).netloc != parse.urlparse(root).netloc:
                    continue
                if link not in visited:
                    queue.put(link)
            counter += 1
            
        except Exception as e:
            print(e, url)

    return visited, extracted


def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    results = []
    h = str(html)
    for match in re.findall(r'(?:\(?\+?1\)?[-.\s]?)?\(?[2-9][0-8][0-9]\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}', h):
        results.append((address, 'PHONE', match))
    for match in re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', h):      
        results.append((address, 'EMAIL', match))
    for match in re.findall(r'[A-Za-z\s]+,\s+[A-Za-z.\s]+(?:,?\s+[A-Za-z\s]+)?\s+\d{5}(?:-\d{4})?', h):    
        results.append((address, 'ADDRESS', match))  
    return results


def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)


def main():
    site = sys.argv[1]

    links = get_links(site)
    writelines('links.txt', links)

    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)

    visited, extracted = crawl(site)
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)

if __name__ == '__main__':
    main()