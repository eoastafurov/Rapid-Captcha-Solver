import re 
import os
import json

SOURCE = '/Volumes/Samsung_T5/DatasetHCAPTCHA/google-images-download/images'
SUPPORTED_IMG_TYPES = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']

def extract_class_name_from_dname(dname: str):
    return re.match(
        pattern='\[(.*?)\]', 
        string=dname
        ).group(0)[:-1][1:]


def load_fnames():
    result = {}
    for dname in os.listdir(SOURCE):
        adname = os.path.join(SOURCE, dname)
        if not os.path.isdir(adname):
            continue
        
        class_name = extract_class_name_from_dname(dname)
        if not class_name in result:
            result[class_name] = []
        
        for fname in os.listdir(adname):
            if not fname.split('.')[-1] in SUPPORTED_IMG_TYPES:
                continue
            afname = os.path.join(adname, fname)
            result[class_name].append(afname)

    with open('./tmp/fnames.json', 'w+') as f:
        json.dump(result, f, indent=4)
    
    return result
