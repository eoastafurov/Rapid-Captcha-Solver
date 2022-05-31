import os
from pydoc import classname
import re 
import hashlib
import cv2
from tqdm import tqdm 
import json


def extract_class_name_from_dname(dname: str):
    return re.match(
        pattern='\[(.*?)\]', 
        string=dname
        ).group(0)[:-1][1:]


def read(buff_path):
    unique = set()
    paths = {}
    counter = 0
    try:
        pbar = tqdm(total=len(os.listdir(buff_path)))
        for dname in os.listdir(buff_path):
            adname = os.path.join(buff_path, dname)
            if not os.path.isdir(adname):
                continue
            class_name = extract_class_name_from_dname(dname)
            if not class_name in paths:
                paths[class_name] = []
            pbar.set_description(class_name)
            for fname in os.listdir(adname):
                pbar.set_postfix({'total': counter})
                afname = os.path.join(adname, fname)
                img = cv2.imread(afname)
                if img is None:
                    continue
                im_id = hashlib.md5(img).hexdigest()
                if not im_id in unique:
                    unique.add(im_id)
                    paths[class_name].append(afname)
                    counter += 1
            pbar.update(1)
                
    except KeyboardInterrupt:
        pbar.close()
        print('Graceful shutdown')
        return paths, unique
    
    return paths, unique
        

def save(paths, savedir='UNIQUE_BUFF'):
    for idx, class_name in enumerate(paths.keys()):
        adname = os.path.join(savedir, class_name)
        if not os.path.exists(adname):
            os.mkdir(adname)
        pbar = tqdm(total=len(paths[class_name]), desc='{}/{}'.format(idx + 1, len(paths.keys())))
        for i, source_afname in enumerate(paths[class_name]):
            img = cv2.imread(source_afname)
            dest_afname = os.path.join(adname, '{}.jpg'.format(i))
            cv2.imwrite(dest_afname, img)
            pbar.update(1)
        pbar.close()
            
        

def process():
    paths, unique = read('./images/BUFF')
    with open('deleteme.json', 'w+') as f:
        json.dump(paths, f, indent=4)
    print('Unique len: {}'.format(len(unique)))
    save(paths=paths)



if __name__ == '__main__':
    process()
