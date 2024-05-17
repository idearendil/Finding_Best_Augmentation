from random import random
from train import test_aug
from aug_move import prepare_dataset
import pickle
import csv

RADICAL_MUTRATE = 0.1
GRADUAL_MUTRATE = 0.3
GRADUAL_MUT_VEL = 0.05

LIMIT_DICT = {
    'crop_rate': [0, 0.5, 0.5],
    'rotate': [0, 90, 90],
    'p_rotate': [0, 1, 1],
    'shear': [0, 45, 45],
    'p_shear': [0, 1, 1],
    'perspective': [0, 1, 1],
    'p_perspective': [0, 1, 1],
    'hsv_h': [0, 0.5, 0.5],
    'hsv_s': [0, 1, 1],
    'hsv_v': [0, 1, 1],
    'p_hsv': [0, 1, 1],
    'motion_blur_limit': [1, 100, 99],
    'p_motion_blur': [0, 1, 1],
    'gaussian_blur_limit': [1, 99, 98],
    'p_gaussian_blur': [0, 1, 1],
    'gaussian_noise_var_limit': [0, 100, 100],
    'p_gaussian_noise': [0, 1, 1],
}


def radical_mutation(a_key):
    rnd_value = random() * LIMIT_DICT[a_key][2] + LIMIT_DICT[a_key][0]
    if a_key == 'motion_blur_limit' or a_key == 'gaussian_blur_limit':
        rnd_value = (rnd_value // 2) * 2 + 1
        rnd_value1 = min(max(1, rnd_value), 99)
        rnd_value = random() * LIMIT_DICT[a_key][2] + LIMIT_DICT[a_key][0]
        rnd_value = (rnd_value // 2) * 2 + 1
        rnd_value2 = min(max(1, rnd_value), 99)
        rnd_value = [int(min(rnd_value1, rnd_value2)), int(max(rnd_value1, rnd_value2))]
    elif a_key == 'gaussian_noise_var_limit':
        rnd_value1 = rnd_value
        rnd_value2 = random() * LIMIT_DICT[a_key][2] + LIMIT_DICT[a_key][0]
        rnd_value = [min(rnd_value1, rnd_value2), max(rnd_value1, rnd_value2)]
    return rnd_value

def gradual_mutation(pre_value, a_key):
    if isinstance(pre_value, list):
        ret_value = []
        
        rnd_value = random()
        if rnd_value < 0.5:
            ret_value.append(pre_value[0] - LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL)
        else:
            ret_value.append(pre_value[0] + LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL)
        
        rnd_value = random()
        if rnd_value < 0.5:
            ret_value.append(pre_value[1] - LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL)
        else:
            ret_value.append(pre_value[1] + LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL)
            
        if a_key == 'motion_blur_limit' or a_key == 'gaussian_blur_limit':
            ret_value[0] = int((ret_value[0] // 2) * 2 + 1)
        if a_key == 'motion_blur_limit' or a_key == 'gaussian_blur_limit':
            ret_value[1] = int((ret_value[1] // 2) * 2 + 1)
        
        ret_value[0] = min(max(LIMIT_DICT[a_key][0], ret_value[0]), LIMIT_DICT[a_key][1])
        ret_value[1] = min(max(LIMIT_DICT[a_key][0], ret_value[1]), LIMIT_DICT[a_key][1])
        
        return [min(ret_value), max(ret_value)]
    
    rnd_value = random()
    if rnd_value < 0.5:
        ret_value = pre_value - LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL
    else:
        ret_value = pre_value + LIMIT_DICT[a_key][2] * GRADUAL_MUT_VEL
    ret_value = min(max(LIMIT_DICT[a_key][0], ret_value), LIMIT_DICT[a_key][1])
    return ret_value


def mutation(parent_gene):
    child_gene_lst = []
    for _ in range(10):
        child_gene = {}
        for a_key in parent_gene:
            rnd_value = random()
            if rnd_value < GRADUAL_MUTRATE + RADICAL_MUTRATE:
                if rnd_value < RADICAL_MUTRATE:
                    child_gene[a_key] = radical_mutation(a_key)
                else:
                    child_gene[a_key] = gradual_mutation(parent_gene[a_key], a_key)
            else:
                child_gene[a_key] = parent_gene[a_key]
        child_gene_lst.append(child_gene)
    return child_gene_lst
                
def evaluation(a_gene):
    prepare_dataset(a_gene)
    score = test_aug()
    return score

def run():
    
    parent_gene = {
        'crop_rate' : (0.4 / 1.4),
        'rotate':15,
        'p_rotate':0.7,
        'shear':10,
        'p_shear':0.5,
        'perspective':0.1,
        'p_perspective':0.5,
        'hsv_h':0.05,
        'hsv_s':0.7,
        'hsv_v':0.6,
        'p_hsv':0.5,
        'motion_blur_limit':[7, 51],
        'p_motion_blur':0.7,
        'gaussian_blur_limit':[9, 55],
        'p_gaussian_blur':0.95,
        'gaussian_noise_var_limit':[10.0, 90.0],
        'p_gaussian_noise':0.25
    }
    
    min_score = evaluation(parent_gene)
    
    while True:
        
        with open('best.pickle','wb') as fw:
            pickle.dump(parent_gene, fw)
        
        child_gene_lst = [parent_gene]
        child_gene_lst = child_gene_lst + mutation(parent_gene)
        score_lst = [min_score]
        
        for a_gene in child_gene_lst:
            score_lst.append(evaluation(a_gene))
        
        min_score = min(score_lst)
        f = open("log.csv", "a")
        writer = csv.writer(f)
        writer.writerow(score_lst + [min_score])
        f.close()
        
        flag = False
        for i, a_gene in enumerate(child_gene_lst):
            if score_lst[i] == min_score:
                flag = True
                parent_gene = a_gene
        if not flag:
            print("error! error! error!")
            return


if __name__ == '__main__':
    run()
