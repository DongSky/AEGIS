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
You are an intelligent assistant capable of answering questions in a format that combines text and images (responses include both text and images).

Since you cannot directly generate images, please use <image> in the output string to represent the image. We will later use a T2I model for post-processing, replacing <image> with the generated image. The <image> tag needs to be placed appropriately to ensure it fits the context.

Based on this, provide an interleaved output for the given question.\n
"""


t2i_system_prompt = """
We need you to generate an image to replace the '<dst_img>' tag in the input prompt based on the context. 
The '<dst_img>' represents the image you need to output, while '<src_img>' refers to the input image tags within the prompt, which correspond to images that have already been provided to you in sequential order. 
You need to understand the semantics of the original prompt and create an image that corresponds to the '<dst_img>' tag. This image should maintain stylistic consistency with the other images and align seamlessly with the original prompt's content without any sense of incongruity.

Here is the input prompt:\n
"""

total_save_dir = None


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description="interleave evaluation")
    parser.add_argument('--input-json', default='data/annotation.json')
    parser.add_argument('--model-name', type=str, default='gpt-4o')
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

    prompt = item['prompt']
    input_image = item['input_image']
    cur_id = item['id']
    output_image_dir = f'{total_save_dir}/output_images/{cur_id}'

    to_write_item = copy.deepcopy(item)
    os.makedirs(f'{total_save_dir}/split_results', exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)


    output_json_file = f'{total_save_dir}/split_results/{cur_id}.json'
    if os.path.exists(output_json_file):
        log_dict = {}
        log_dict['id'] = cur_id
        log_dict['is_successful'] = True
        log_dict['is_text_successful'] = True
        log_dict['is_image_successful'] = True
        return log_dict

    parts = re.split(r'(<image\d+>)', prompt)
    prompt_part_list = [part if not re.match(r'<image\d+>', part) else '<image>' for part in parts]

    input_data = []
    image_id = 0

    for part in prompt_part_list:
        part = part.strip()

        if part == '<image>':
            image_path = os.path.join(image_dir, input_image[image_id])
            base64_image = image_to_base64(image_path)
            input_data.append({'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{base64_image}"}})
            image_id += 1
        else:
            if len(part) != 0 and part != ' ':
                input_data.append({'type': 'text', 'text': part})

    num_has_tries = 0
    response = None
    is_text_successful = False

    while num_has_tries <= num_max_tries:
        try:
            response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_data}
                    ]
                )
            output_text = response.choices[0].message.content
            is_text_successful = True
            break
        except Exception as e:
            print(f'{cur_id} text generation failed, retry for {e}')
            output_text = 'failed'
            num_has_tries += 1
            continue

    to_write_item['output_text'] = output_text
    to_write_item['model_name'] = 'gpt-4o'

    if not is_text_successful:
        print(f'{cur_id} is text generation failed')
        log_dict = {}
        log_dict['id'] = cur_id
        log_dict['is_text_successful'] = False
        log_dict['is_image_successful'] = False
        return log_dict

    question_parts = re.split(r'(<image\d+>)', prompt)
    question_prompt_part_list = [part if not re.match(r'<image\d+>', part) else '<src_img>' for part in question_parts]
    lines = []
    for item in question_prompt_part_list:
        if len(item) != 0:
            lines.append(item)
    question_str = ''.join(lines) + '\n'

    output_id = 0
    cnt_image_tag = 0
    num_output_image = output_text.count('<image>')
    to_write_item['num_output_image'] = num_output_image
    ans_parts = re.split(r'(<image>)', output_text)
    cur_prompt = t2i_system_prompt + question_str
    output_image_path_list = []

    is_image_successful = True

    for idx, ans_part in enumerate(ans_parts):

        num_has_tries = 0
        is_sub_image_successful = False

        if ans_part != '<image>':
            cur_prompt += ans_part
            continue

        cnt_image_tag += 1
        if cnt_image_tag == num_output_image:
            remain_str = ''.join(ans_parts[idx:]).replace('<image>', '')
            cur_prompt_for_use = cur_prompt + '\n' + '<dst_img>' + remain_str
        else:
            cur_prompt_for_use = cur_prompt + '\n' + '<dst_img>'

        input_image_rb_list = []
        for item in input_image:
            if 'output-' not in item:
                cur_item = os.path.join(image_dir, item)
            else:
                cur_item = item
            input_image_rb_list.append(open(cur_item, 'rb'))

        while num_has_tries <= num_max_tries:
            try:

                if image_id == 0:
                    result = client.images.generate(
                        model='gpt-image-1',
                        prompt=cur_prompt_for_use,
                        quality='medium'
                    )
                else:
                    result = client.images.edit(
                        model='gpt-image-1',
                        image=input_image_rb_list,
                        prompt=cur_prompt_for_use,
                        quality='medium'
                    )
                is_sub_image_successful = True
                break
            except Exception as e:
                print(f'{cur_id} image generation for {idx}/{cnt_image_tag} failed, retry')
                num_has_tries += 1
                continue

        if not is_sub_image_successful:
            is_image_successful = False
            break
                
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        dst_image_path = f"{output_image_dir}/output-{cur_id}-{output_id}.jpg"

        with open(dst_image_path, "wb") as f:
            f.write(image_bytes)

        input_image.append(dst_image_path)
        output_image_path_list.append(dst_image_path)
        output_id += 1
        image_id += 1

        cur_prompt = cur_prompt + '\n' + '<src_img>' + '\n'

    to_write_item['output_image_path_list'] = output_image_path_list

    if is_image_successful:

        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(to_write_item, f, ensure_ascii=False, indent=4)

    log_dict = {}
    log_dict['id'] = cur_id
    log_dict['is_text_successful'] = is_text_successful
    log_dict['is_image_successful'] = is_image_successful
    log_dict['is_successful'] = is_text_successful and is_image_successful

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
    sub_output_dir = f'{output_dir}/interleaved'


    os.makedirs(sub_output_dir, exist_ok=True)

    print(f'{model_name=}, {input_json=}, {output_dir=}')

    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        if item['task'] == 'interleaved':
            data.append(item)

    eval_model(data, model_name, sub_output_dir, num_processes)



if __name__ == "__main__":
    main()