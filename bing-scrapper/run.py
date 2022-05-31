from configparser import DuplicateOptionError
import os 
from multiprocessing import Process
import json

def run(query: str):
    os.system(query)

def process():
    with open('urls.json', 'r') as f:
        dictionary = json.load(f)

    for class_name in dictionary['dictionary'].keys():
        processes = []
        for elem in dictionary['dictionary'][class_name]:
            if elem['processed']:
                continue
            s_query = "+".join(elem['query'].strip().split(' '))
            query = 'python3 bing_scraper.py  --url "https://www.bing.com/images/search?q={}" '.format(s_query)
            query += '--limit 1000 --download --chromedriver /Users/evgenijastafurov/Downloads/chromedriver '
            query += '--image_directory "./BUFF/[{}] {}"'.format(class_name, elem['query'])
            print('Starting proc with query: {}'.format(elem['query']))
            proc = Process(target=run, args=(query,)) 
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
            
        for elem in dictionary['dictionary'][class_name]:
            if not elem['processed']:
                elem['processed'] = True
        
    with open('urls=.json', 'w+') as f:
        json.dump(dictionary, f, indent=4)           
                       
            
            
        
        
if __name__ == '__main__':
    process()





# if __name__ == '__main__':
#     processes = []
#     for key in urls.keys():
        # query = 'python3 bing_scraper.py  --url "https://www.bing.com/images/search?q={}" '.format(key)
        # query += '--limit 1000 --download --chromedriver /Users/evgenijastafurov/Downloads/chromedriver '
        # query += '--image_directory "[{}] {}"'.format(urls[key], key)
#         print(query)
#         proc = Process(target=run, args=(query,))
#         processes.append(proc)
#         proc.start()

    # for proc in processes:
    #     proc.join()





























# urls = {
#     'CLA+250+AMG': 'car',
#     'bmw+3': 'car',
#     'car': 'car',
#     'Ford+Cars': 'car',
#     'Small+Cars': 'car',
#     'Best+Compact+Cars': 'car',
#     'Car+Drawing': 'car',
#     'muscle+cars': 'car',
#     'big+cars': 'car',
#     'Big+Luxury+Cars': 'car',
#     'Large+Cars': 'car',
#     'Family+Car': 'car',
#     'Kia+City+Car': 'car',
#     'car+India': 'car'
# }