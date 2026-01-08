import json
import argparse
from collections import defaultdict
import numpy as np

task_list = ['understanding', 'editing', 'generation', 'interleaved']
stem_list = ['biology', 'chemistry', 'mathematics', 'medicine', 'physics', 'astronomy_geography', 'it'] 
humanities_list = ['agriculture', 'history', 'movie', 'music', 'art', 'culture', 'architecture']
daily_list = ['activity', 'anime', 'game', 'traffic', 'photography', 'food', 'engineering']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str)
    args = parser.parse_args()
    return args


def main():
    eval_failed_list = []
    infer_failed_list = []

    args = parse_args()
    input_json = args.input_json
    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    high_topic_score_dict = defaultdict(list)
    low_topic_score_dict = defaultdict(list)

    for item in data:
        low_topic = item['category']
        cur_id = item['id']
        output_text = item['output_text']
        if 'answer_list' not in item:
            eval_failed_list.append(cur_id)
            continue

        if output_text == 'failed':
            infer_failed_list.append(cur_id)
            continue

        answer_list = item['answer_list']
        refined_question = item['refined_question']

        if len(answer_list) != len(refined_question):
            eval_failed_list.append(cur_id)
            continue

        if low_topic in stem_list:
            high_topic = 'stem'
        elif low_topic in humanities_list:
            high_topic = 'humanities'
        elif low_topic in daily_list:
            high_topic = 'daily'
        else:
            raise ValueError(f'Invalid category: {low_topic} for {cur_id}')

        cur_right = 0

        for cur_ans in answer_list:
            if cur_ans in ['Y', 'y', 'yes', 'Yes', 'YES']:
                cur_right += 1

        cur_score = float(cur_right) / len(answer_list)

        high_topic_score_dict[high_topic].append(cur_score)
        low_topic_score_dict[low_topic].append(cur_score)

    for high_topic, score_list in high_topic_score_dict.items():
        print(f'{high_topic=}: {np.mean(score_list):.4f}, {np.std(score_list):.4f}')
    for low_topic, score_list in low_topic_score_dict.items():
        print(f'{low_topic=}: {np.mean(score_list):.4f}, {np.std(score_list):.4f}')
    print(f'{len(eval_failed_list)=}, {len(infer_failed_list)=}')

if __name__ == '__main__':
    main()