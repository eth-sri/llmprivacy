import bs4
import json
from bs4 import BeautifulSoup

with open('c2ssoccup.shtml', 'r') as fin:
    html_doc = fin.read()

soup = BeautifulSoup(html_doc, 'html.parser')

table = soup.find("table")

mapping = {}

for tr in table.contents[-1].contents:
    if isinstance(tr, bs4.element.Tag):
        if len(tr.contents) == 13:
            code = tr.contents[9].contents[0]
            occ_name = tr.contents[11].contents[0]
            if code.isnumeric():
                code = int(code)
                mapping[code] = occ_name


print(mapping)
with open('occ_mapping.json', 'w') as fou:
    json.dump(mapping, fou)
