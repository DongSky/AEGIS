import json
from tqdm import tqdm
import argparse
import re
import os

import base64
from google import genai
from PIL import Image
from io import BytesIO

import copy
from multiprocessing import Pool
from google.genai.types import HttpOptions


image_dir = 'data/images'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TIMEOUT = 3 * 60 * 1000 # 3 minutes
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
num_max_tries = 5



def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def parse_args():
    parser = argparse.ArgumentParser(description="editing evaluation")
    parser.add_argument('--input-json', default='data/annotation.json')
    parser.add_argument('--model-name', type=str, default='gemini-2.5-flash-image')
    parser.add_argument('--output-dir', type=str, default='output-gemini')
    parser.add_argument('--num-processes', type=int, default=64, help="Number of processes to use")
    args = parser.parse_args()
    return args


system_prompt = """
You are an intelligent assistant capable of answering questions by generating a single image.

Based on this, provide a single image output for the given question. The question may contain some images in addition to the original image which is needed to be edited.

Noted that the output image should be a single image!

Please answer the question:\n
"""


def process_item(item):

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(base_url=GEMINI_BASE_URL, timeout=GEMINI_TIMEOUT)
    )

    prompt = item['prompt']
    input_image = item['input_image']
    cur_id = item['id']
    output_image_dir = f'{total_save_dir}/output_images'

    os.makedirs(f'{total_save_dir}/split_results', exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    output_json_file = f'{total_save_dir}/split_results/{cur_id}.json'

    if os.path.exists(output_json_file):
        print(f'{cur_id} already exists, continue')
        log_dict = {}
        log_dict['id'] = cur_id
        log_dict['is_successful'] = True
        return log_dict

    to_write_item = copy.deepcopy(item)
    parts = re.split(r'(<image\d+>)', prompt)
    prompt_part_list = [part if not re.match(r'<image\d+>', part) else '<image>' for part in parts]

    input_data = []
    image_id = 0

    input_data.append(system_prompt)

    for part in prompt_part_list:
        part = part.strip()

        if part == '<image>':
            image_path = os.path.join(image_dir, input_image[image_id])
            cur_image = Image.open(image_path)
            image_id += 1
            input_data.append(cur_image)
        else:
            if len(part) != 0 and part != ' ':
                input_data.append(part)

    num_has_tries = 0
    is_successful = False
    output_image_path_list = []

    while num_has_tries <= num_max_tries:
        try:
            response = client.models.generate_content(
                model=args.model_name,
                contents=input_data
            )

            output_text = ''
            output_id = 0

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    output_text += part.text
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    dst_img_path = f"{output_image_dir}/{cur_id}-{output_id}.jpg"
                    image.save(dst_img_path)
                    output_id += 1
                    output_image_path_list.append(dst_img_path)
                    output_text += '\n' + '<image>' + '\n'
            is_successful = True
            break
        except Exception as e:
            print(f'{cur_id} failed, retry for {e}')
            num_has_tries += 1
            continue

    if not is_successful:
        output_text = 'failed'
        output_image_path_list = []

    to_write_item['output_image_path_list'] = output_image_path_list
    to_write_item['output_text'] = output_text
    to_write_item['model_name'] = args.model_name
    print(f'{output_text=}')


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
    output_image_dir = f'{output_dir}/output_images'
    os.makedirs(output_image_dir, exist_ok=True)

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
    sub_output_dir = f'{output_dir}/editing'

    os.makedirs(sub_output_dir, exist_ok=True)

    print(f'{model_name=}, {input_json=}, {output_dir=}')

    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        if item['task'] == 'editing':
            data.append(item)

    eval_model(data, model_name, sub_output_dir, num_processes)



if __name__ == "__main__":
    main()