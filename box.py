"""
curl https://api.box.com/2.0/shared_items -H "Authorization: Bearer L3InhL8dvKvyuztrfeuDsASnkd8wLntk" -H "BoxApi: shared_link=https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019/folder/79732510815"
curl https://api.box.com/2.0/folders/72677118496/items -H "Authorization: Bearer L3InhL8dvKvyuztrfeuDsASnkd8wLntk" -H "BoxApi: shared_link=https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019"
curl https://api.box.com/2.0/folders/79732510815/items -H "Authorization: Bearer L3InhL8dvKvyuztrfeuDsASnkd8wLntk" -H "BoxApi: shared_link=https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019"
curl https://api.box.com/2.0/folders/79729844333/items -H "Authorization: Bearer L3InhL8dvKvyuztrfeuDsASnkd8wLntk" -H "BoxApi: shared_link=https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019"
curl https://api.box.com/2.0/files/477815866270/content -H "Authorization: Bearer L3InhL8dvKvyuztrfeuDsASnkd8wLntk" -H "BoxApi: shared_link=https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019"
"""
from boxsdk import OAuth2, Client
#from boxsdk import Folder

auth = OAuth2(
    client_id='jvy2fowomvxmta7m460nrlyhds2f6h3g',
    client_secret='kUHuouH9jdYkq7zXRQBWdZhBxmK7VfJ3',
    access_token='L3InhL8dvKvyuztrfeuDsASnkd8wLntk'
)
client = Client(auth)

"""
root_folder = client.root_folder().get()

items = root_folder.get_items()
for item in items:
    print('{0} {1} is named "{2}"'.format(item.type.capitalize(), item.id, item.name))
    with open(item.name, 'wb') as open_file:
        client.file(item.id).download_to(open_file)
        open_file.close()
"""
#with open('test.tar.gz', 'wb') as open_file:
#    client.file(file_id='477815866270').download_to(open_file)
#    open_file.close()

shared_client = client.with_shared_link('https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019', '')
#shared_folder = shared_client.get_shared_item(SHARED_LINK_URL)

#folder_contents = shared_folder.get_items()
subfolder = shared_client.folder(79729844333).get()
for item in subfolder.get_items(limit=1000):
    client.file(file_id=item.id).content()
