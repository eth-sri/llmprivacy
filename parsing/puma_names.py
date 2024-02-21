# importing required modules
import PyPDF2
import json
  
# creating a pdf file object
pdfFileObj = open('2010_PUMA_Names.pdf', 'rb')
  
# creating a pdf reader object
reader = PyPDF2.PdfReader(pdfFileObj)
  
# printing number of pages in pdf file

puma_mapping = [{} for _ in range(100)]
  
# extracting text from page
for page_idx in range(len(reader.pages)):
    page_obj = reader.pages[page_idx]
    for line in page_obj.extract_text().split('\n'):
        if len(line) < 9:
            continue
        if line == '2010 PUMA Names File' or line == 'January 2014' or line == '2010 PUMA Names FileSTATEFP PUMA5CE PUMA NAME':
            continue
        print(line)
        
        state = int(line[:2])
        puma_code = int(line[3:8])
        puma_name = line[9:]

        puma_mapping[state][puma_code] = puma_name
        
# closing the pdf file object
pdfFileObj.close()

with open('puma_mapping.json', 'w') as fou:
    json.dump(puma_mapping, fou)
