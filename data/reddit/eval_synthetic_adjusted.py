import json
import numpy as np


features = [
    'age', 'sex', 'city_country', 'birth_city_country', 'education', 'occupation', 'income_level', 'relationship_status'
]

hardness = 5

def eval_split(split, features, hardness):
    eval_dict = {feature: {hard+1: [] for hard in range(hardness)} for feature in features}
    for sample in split:
        eval_dict[sample['feature']][sample['hardness']].append(sample['guess_correctness']['model_aided_eval'])
    return eval_dict

def print_eval(eval_dict, hardness):
    per_hardness = {h+1: [] for h in range(hardness)}
    all_data = []
    for feature, feature_data in eval_dict.items():
        print(feature)
        all_hardness_feature_data = []
        for hardness, hardness_data in feature_data.items():
            all_hardness_feature_data.extend(hardness_data)
            per_hardness[hardness].extend(hardness_data)
            all_data.extend(hardness_data)
            if len(hardness_data) < 1:
                to_print = str(hardness) + ':'
            else:
                to_print = str(hardness) + ':   ' + '    '.join([f'{100*acc:.1f}%' for acc in np.mean(hardness_data, axis=0)])
            print(to_print)
        overall_feature_print = 'Ovr.:' + '    '.join([f'{100*acc:.1f}%' for acc in np.mean(all_hardness_feature_data, axis=0)])
        print(overall_feature_print)
        print('\n')
    
    print('All:  ' + '    '.join([f'{100*acc:.1f}' for acc in np.mean(all_data, axis=0)]))

    print('\nHardness:')
    for h, hdata in per_hardness.items():
        print(str(h) + ':   ' + '    '.join([f'{100*acc:.1f}%' for acc in np.mean(hdata, axis=0)]))



path1 = 'samples_all_hardness_all_features_0_0_split1_adjusted.jsonl'
path2 = 'samples_all_hardness_all_features_0_0_split2_adjusted.jsonl'

# load jsons
with open(path1, 'r') as f:
    split1 = [json.loads(line) for line in f]

with open(path2, 'r') as f:
    split2 = [json.loads(line) for line in f]

joint = split1 + split2
print(len(joint))

with open('synthetic_dataset.jsonl', 'w') as f:
    for d in joint:
        f.write(json.dumps(d) + '\n')

joint_eval = eval_split(joint, features, hardness)
split1_eval = eval_split(split1, features, hardness)
split2_eval = eval_split(split2, features, hardness)


print('Joint eval')
print_eval(joint_eval, hardness)
print('\n\n')
print('Split 1 eval')
print_eval(split1_eval, hardness)
print('\n\n')
print('Split 2 eval')
print_eval(split2_eval, hardness)
print('\n\n\n\n')
