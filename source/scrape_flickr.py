from flickrapi import FlickrAPI, shorturl
import threading,requests, os, urllib


API_KEY = "07e391a86e66960a7449ea81d0d9cd48"

flickr = FlickrAPI(API_KEY, API_KEY, format='parsed-json')

def photo_url(info):
    return "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}.jpg".format(**info)

image_urls = []
for p in range(30):
    walk = flickr.photos.search(tags='whiteboard', text='whiteboard', page=p)
    image_urls+=[photo_url(w) for w in walk['photos']['photo']]


target_directory = 'images/'

# In our case we want to download the
# images as fast as possible, so we use threads.
class FetchResource(threading.Thread):
    """Grabs a web resource and stores it in the target directory.

    Args:
        target: A directory where to save the resource.
        urls: A bunch of urls to grab

    """
    def __init__(self, target, urls):
        super().__init__()
        self.target = target
        self.urls = urls

    def run(self):
        for url in self.urls:
            url = urllib.parse.unquote(url)
            with open(os.path.join(self.target, url.split('/')[-1]), 'wb') as f:
                try:
                    content = requests.get(url).content
                    f.write(content)
                except Exception as e:
                    pass
                print('[+] Fetched {}'.format(url))

# make a directory for the results
try:
    os.mkdir(target_directory)
except FileExistsError:
    pass


# fire up 100 threads to get the images
num_threads = 100

threads = [FetchResource('images/', []) for i in range(num_threads)]

print("Got images:", len(image_urls))
while image_urls:
    for t in threads:
        try:
            t.urls.append(image_urls.pop())
        except IndexError as e:
            break

threads = [t for t in threads if t.urls]

for t in threads:
    t.start()

for t in threads:
    t.join()

# that's it :)
