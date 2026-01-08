import json
from tqdm import tqdm
import argparse
import re
import os

import base64
from openai import OpenAI

import copy

from multiprocessing import Pool

image_dir = 'data/images'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_TIMEOUT = 120
num_max_tries = 5

system_prompt = """
You are an intelligent assistant capable of answering questions by generating a single image.

The text of the problem contains special tags such as <image>, and the corresponding images are provided to you in sequence.

Based on this, provide a single image output for the given question without any other text.

Please answer the question:\n
"""

def parse_args():
    parser = argparse.ArgumentParser(description="generation evaluation")
    parser.add_argument('--input-json', default='data/annotation.json')
    parser.add_argument('--model-name', type=str, default='gpt-image-1')
    parser.add_argument('--output-dir', type=str, default='output-4o')
    parser.add_argument('--num-processes', type=int, default=64, help="Number of processes to use")
    args = parser.parse_args()
    return args



def process_item(item):

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT
    )

    num_has_tries = 0
    is_successful = False

    prompt = item['prompt']
    input_image = item['input_image']
    cur_id = item['id']
    output_image_dir = f'{total_save_dir}/output_images'

    to_write_item = copy.deepcopy(item)
    os.makedirs(f'{total_save_dir}/split_results', exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)


    output_json_file = f'{total_save_dir}/split_results/{cur_id}.json'
    if os.path.exists(output_json_file):
        log_dict = {}
        log_dict['id'] = cur_id
        log_dict['is_successful'] = True
        return log_dict

    parts = re.split(r'(<image\d+>)', prompt)
    prompt_part_list = [part if not re.match(r'<image\d+>', part) else '<image>' for part in parts]
    
    image_id = 0

    input_image_rb_list = []

    for part in prompt_part_list:
        part = part.strip()

        if part == '<image>':
            image_path = os.path.join(image_dir, input_image[image_id])
            input_image_rb_list.append(open(image_path, 'rb'))
            image_id += 1
        else:
            pass

    cur_prompt_for_use = ' '.join(prompt_part_list)
    cur_prompt_for_use = system_prompt + cur_prompt_for_use
    result = None

    while num_has_tries <= num_max_tries:
        try:
            if len(input_image_rb_list) == 0:
                result = client.images.generate(
                        model=args.model_name,
                        prompt=cur_prompt_for_use,
                        quality='medium')
            else:
                result = client.images.edit(
                    model=args.model_name,
                    image=input_image_rb_list,
                    prompt=cur_prompt_for_use,
                    quality='medium')
            is_successful = True
            break
        except Exception as e:
            print(f'{cur_id} failed, retry for {e}')
            num_has_tries += 1
            continue

    dst_img_path = f"{output_image_dir}/{cur_id}.jpg"
    if is_successful:
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        with open(dst_img_path, "wb") as f:
            f.write(image_bytes)

        to_write_item['output_text'] = ''
        to_write_item['model_name'] = args.model_name
        to_write_item['output_image_path_list'] = [dst_img_path]

        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(to_write_item, f, ensure_ascii=False, indent=4)
    else:
        to_write_item['output_text'] = 'failed'
        to_write_item['output_image_path_list'] = []
        to_write_item['model_name'] = args.model_name
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(to_write_item, f, ensure_ascii=False, indent=4)

    log_dict = {}
    log_dict['id'] = cur_id
    log_dict['is_successful'] = is_successful

    return log_dict
    


def eval_model(data, model_name, output_dir, num_processes):
    global total_save_dir
    total_save_dir = output_dir

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_item, data), total=len(data)))

    output_log_file = f'{output_dir}/log.json'
    output_json_file = f'{output_dir}/results.json'

    results = sorted(results, key=lambda x: int(x['id']))

    with open(output_log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    failed_id = []
    for item in results:
        if not item['is_successful']:
            failed_id.append(item['id'])

    save_results = []
    for item in data:
        cur_id = item['id']
        if cur_id in failed_id:
            cur_item = copy.deepcopy(item)
            cur_item['output_text'] = 'failed'
            cur_item['output_image_path_list'] = []
            cur_item['model_name'] = model_name
            save_results.append(cur_item)
            continue


        src_json_file = f'{total_save_dir}/split_results/{cur_id}.json'
        with open(src_json_file, 'r', encoding='utf-8') as f:
            src_data = json.load(f)
        save_results.append(src_data)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=4)



def main():
    global args
    args = parse_args()
    model_name = args.model_name
    input_json = args.input_json
    output_dir = args.output_dir
    num_processes = args.num_processes
    sub_output_dir = f'{output_dir}/generation'


    os.makedirs(sub_output_dir, exist_ok=True)

    print(f'{model_name=}, {input_json=}, {output_dir=}')

    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        if item['task'] == 'generation':
            data.append(item)

    eval_model(data, model_name, sub_output_dir, num_processes)



if __name__ == "__main__":
    main()