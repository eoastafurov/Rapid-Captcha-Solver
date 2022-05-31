import os

TARGET_DIR = '/home/eugeny/ibot-ml/Small'

def empty_target_dir():
    if os.path.exists(TARGET_DIR):
        os.system('rm -rf {}'.format(TARGET_DIR))
    os.system('mkdir {}'.format(TARGET_DIR))
    os.system('mkdir {}/train'.format(TARGET_DIR))
    os.system('mkdir {}/test'.format(TARGET_DIR))
    os.system('mkdir {}/val'.format(TARGET_DIR))
    