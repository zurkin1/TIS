from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
import re
import os

for i in range(5,10):
        url = "https://m.box.com/shared_item/https%3A%2F%2Fstonybrookmedicine.app.box.com%2Fs%2F7n9gdy3i6qmm638or7lbxrzzydb1iv9b/browse/79732510815?page=" + str(i+1)
        page = urlopen(url)
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")
        #print(html)
        lines = re.split("\n" ,html, flags=re.MULTILINE)
        numbers = []
        for line in lines:
            match = re.search('id="(\d+)', line)
            if match:
                numbers.append(match.group(1))

        for n in numbers:
            url = "https://m.box.com/shared_item/https%3A%2F%2Fstonybrookmedicine.app.box.com%2Fs%2F7n9gdy3i6qmm638or7lbxrzzydb1iv9b/browse/" + str(n)
            print(url)
            page = urlopen(url)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")
            lines = re.split("\n" ,html, flags=re.MULTILINE)
            for line in lines:
                match = re.search('id="(\d+)', line)
                if match:
                    if os.path.isfile(match.group(1)):
                        continue
                    print(match.group(1))
                    download = 'https://m.box.com/file/' + str(match.group(1)) + '/download?shared_link=https%3A%2F%2Fstonybrookmedicine.app.box.com%2Fs%2F7n9gdy3i6qmm638or7lbxrzzydb1iv9b'
                    print(download)
                    os.system('wget -O ' + str(match.group(1)) + ' ' + download )

# url = 'https://m.box.com/shared_item/https%3A%2F%2Fstonybrookmedicine.app.box.com%2Fs%2F7n9gdy3i6qmm638or7lbxrzzydb1iv9b/browse/' + str(79729844333)	
# page = urlopen(url)
# html_bytes = page.read()
# html = html_bytes.decode("utf-8")
# print(html)
# lines = re.split("\n" ,html, flags=re.MULTILINE)
# for line in lines:
# 	match = re.search('id="(\d+)', line)
# 	if match:
# 		print(match.group(1))	
# 		download = 'https://m.box.com/file/' + str(match.group(1)) + '/download?shared_link=https%3A%2F%2Fstonybrookmedicine.app.box.com%2Fs%2F7n9gdy3i6qmm638or7lbxrzzydb1iv9b'
# 		print(download)
# 		os.system('wget ' + download )
