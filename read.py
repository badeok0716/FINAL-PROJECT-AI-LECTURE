import pickle
import numpy as np
import imageio

def read_pkl(path, encoding='ASCII', print_option=True):
    '''read path(pkl) and return files
    Dependency : pickle
    Args:
        path - string
               ends with pkl
    Return:
        pickle content
    '''
    if print_option:
        print("Pickle is read from %s"%path)
    with open(path, 'rb') as f: return pickle.load(f, encoding=encoding)

def f(l):
    l.sort()
    ll = l[-5:]
    return sum(ll)/len(ll)

def get_info(mats):
    bleu2, bleu3, bleu4 = [], [] , []
    sbleu2, sbleu3, sbleu4 = [], [], []
    nll = []
    for m in mats:
        bleu2.append(m[0][0])
        bleu3.append(m[0][1])
        bleu4.append(m[0][2])
        sbleu2.append(m[3][0])
        sbleu3.append(m[3][1])
        sbleu4.append(m[3][2])
        nll.append(m[2])
    bleu2 = f(bleu2)
    bleu3 = f(bleu3)
    sbleu2 = f(sbleu2)
    return f'&{bleu2:.3f}&{bleu3:.3f}&{sbleu2:.3f}\\\\'

root = '/home/deokjae/TextGAN-PyTorch/save/aug_exp_final/image_coco/'
import sys
method = 'relgan'
for ratio in [0.05,1.0]:
    print(ratio)
    for aug in ['noaug','swap','rand','swap_rand','mask','swap_rand_mask']:
        mmat = []
        for trial in [1,2,3,4,5,6,7,8,9,10]:
            path = root + '{}/{}_all/{}_temp100_trial{}/content.pkl'.format(method,aug,ratio,trial)
            content = read_pkl(path,print_option=False)
            m = 0
            #print(content)
            mbleu = 0
            mkey = None
            for key in content:
                pass
            mmat.append(content[key])
        print('&',aug, get_info(mmat))

method = 'seqgan'
for ratio in [1.0]:
    print(ratio)
    for aug in ['noaug','swap','rand','swap_rand','mask','swap_rand_mask']:
        mmat = []
        for trial in [1,2,3,4,5,6,7,8,9,10]:
            path = root + '{}/{}_None/{}_temp1_trial{}/content.pkl'.format(method,aug,ratio,trial)
            content = read_pkl(path,print_option=False)
            m = 0
            #print(content)
            mbleu = 0
            mkey = None
            for key in content:
                pass
            mmat.append(content[key])
        print('&',aug, get_info(mmat))

method = 'relgan'
for ratio in [1.0]:
    print(ratio)
    for aug_place in ['all','D_only','D_only2','real_only']:
        for aug in ['swap','rand','mask']:
            mmat = []
            for trial in [1,2,3,4,5,6,7,8,9,10]:
                path = root + '{}/{}_{}/{}_temp100_trial{}/content.pkl'.format(method,aug,aug_place,ratio,trial)
                content = read_pkl(path,print_option=False)
                m = 0
                #print(content)
                mbleu = 0
                mkey = None
                for key in content:
                    pass
                mmat.append(content[key])
            print('&',aug, aug_place, get_info(mmat))