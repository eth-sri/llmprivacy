import json
import numpy as np
import os


hardnesses = 5
persons = 40
samples = 1
features = [
    'age', 'sex', 'city_country', 'birth_city_country', 'education', 'occupation', 'income_level', 'relationship_status'
]
longest = max([len(s) for s in features])

for hardness in range(hardnesses):
    print('\n')
    print(f'Hardness: {hardness+1}')
    overall = []
    for feature in features:

        person_data = []
        for person in range(persons):

            for sample in range(samples):
                path = f'{feature}/{feature}_hard{hardness+1}_pers{person+1}_{sample}.json'

                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    person_data.append([float(err) for err in data['guess_correctness']['model_aided_eval']])

                else:
                    continue
        
        if len(person_data) > 0:
            mean_accs_feature = np.mean(np.array(person_data), axis=0)
            overall.append(mean_accs_feature)
            spaces = ''.join([' ' for _ in range(longest + 3 - len(feature))])
            print(f'{feature}:{spaces}Top1: {mean_accs_feature[0]*100:.1f}%    Top3: {mean_accs_feature[1]*100:.1f}%    Top1 less-precise: {mean_accs_feature[2]*100:.1f}%    Top3 less-precise: {mean_accs_feature[3]*100:.1f}%')

    if len(overall) > 0:
        mean_accs = np.mean(np.array(overall), axis=0)
        spaces = ''.join([' ' for _ in range(longest + 3 - len('Overall'))])
        print(f'Overall:{spaces}Top1: {mean_accs[0]*100:.1f}%    Top3: {mean_accs[1]*100:.1f}%    Top1 less-precise: {mean_accs[2]*100:.1f}%    Top3 less-precise: {mean_accs[3]*100:.1f}%')
