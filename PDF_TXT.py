# This file is to convert PDF file into TXT file
# When you use it, please reassign the input file address and output file address


from tika import parser

"""
raw = parser.from_file('/Users/jinfeng/Downloads/ICO Rating/wp_classified/1m/AB-CHAIN.pdf')

print(raw['content'])

with open('/Users/jinfeng/Downloads/AB-CHAIN.txt' , 'w') as f:
    f.write(raw['content'])
"""


import os
'set up the direction of PDF file'
rootDir = "/Users/jinfeng/Documents/Research Project/ICO Rating/ICO PDF/wp_classified/no-m"

i = 0

for lists in os.listdir(rootDir):
    if not lists.startswith('.'):
        i = i + 1
        path = os.path.join(rootDir,lists)
        raw = parser.from_file(path) # transform the content of pdf to txt
        pdf_name = os.path.basename(path) # get the name of the pdf
        print(pdf_name)
        name = os.path.splitext(pdf_name)[0]
        print(path)
        'The dirction of new txt file should be same to the rootDir'
        with open('/Users/jinfeng/Documents/Research Project/ICO Rating/ICO PDF/txt/no-m/' + name, 'w') as f:
            f.write(raw['content'])

print(i)

"""
import os
rootDir = "/Users/jinfeng/Downloads/ICO Rating/wp_classified/1m"

list_dirs = os.walk(rootDir)

for root, dirs, files in list_dirs:
    for d in dirs:
        print(os.path.join(root,d))
    for f in files:
        path = os.path.join(root,f)
        raw = parser.from_file(path)
        name = os.path.basename(path)
        print(os.path.splitext(name)[0])
"""

