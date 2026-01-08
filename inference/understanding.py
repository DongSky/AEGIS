import json
from tqdm import tqdm
import argparse
import re
import os
from google import genai

from PIL import Image
import copy
from multiprocessing import Pool
from google.genai.types import HttpOptions


eval_prompt = """
**Task: Comprehensive Task-Aware Evaluation**

**Objective:**
You are an expert for world-knowledge-based evaluation. Your task is to verify whether the output image or text meets a series of checklists and provide your reasoning for the evaluation.

**INPUT FORMAT:**
You will be provided with the following fields:
- **Task Type**: The category of the task, which dictates the output modality: `understanding`, `editing`, `generation`, or `interleaved`.
- **Category**: The broader topic for contextual relevance.
- **Clear Prompt**: The original instruction given to the model to output the image or text or both.
- **Reference Text**: The **primary source of truth** for text-based facts.
- **Output Text**: The output texts needed to be evaluated by you.
- **Checklist**: A series of checklists for the output. Each item in the checklist contains one of the tags: `[Text]`, `[Image]`, and `[Consistency]`.

For item with tag `[Text]`, Please focus on analyzing whether the **Output Text** meets the requirements of the checklist item. You can treat the **Reference Text** as a reference that perfectly meets all checklist items. However, the output does not have to be identical to the **Reference Text**. As long as it meets the requirements of the checklist item, it can be marked as passing (Y).


**TASK & OUTPUT REQUIREMENTS:**
Your output must be a single valid JSON object. The Json object should be dict with the following keys:
- **Answer List**: A list of answers for the checklist. Each item in the answer list corresponds to an item in the checklist in order. Each entry is either "Y" or "N," representing "yes" or "no," respectively.
- **Reason List**: The reasoning for the evaluation. Each item in the Reason List explains the reason for the corresponding Y/N in the Answer List.


**Final JSON Output Structure:**
Your entire response must be a single, valid JSON object matching the schema below. Do not include any text outside of this JSON object. It should be noted that the number of elements in the answer_list and reason_list must be the same as the number of items in the Checklist.

```json
{{
    "answer_list": ["Y", "N", "Y", "N", ...],
    "reason_list": ["string1", "string2", "string3", "string4", ...],
}}
```

**INPUT DATA:**
- **Task Type**: {task_type}
- **Category**: {category}
- **Clear Prompt**: {clear_prompt}
- **Reference Text**: {reference_text}
- **Output Text**: {output_text}
- **Checklist**: {checklist}

Evaluate the output according to all requirements above. Ensure the output is valid JSON.
"""

def replace_image_tags(match):
    number = match.group(1)
    if number == '1':
        return 'the first reference image'
    elif number == '2':
        return 'the second reference image'
    elif number == '3':
        return 'the third reference image'
    elif number == '4':
        return f'the {number}th reference image'
    elif number == '5':
        return f'the {number}th reference image'
    else:
        return f'the {number}th reference image'



image_dir = 'data/images'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TIMEOUT = 3 * 60 * 1000 # 3 minutes
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
num_max_tries = 5


total_save_dir = None


def parse_args():
    parser = argparse.ArgumentParser(description="understanding evaluation")
    parser.add_argument('--input-json', default='data/annotation.json')
    parser.add_argument('--model-name', type=str, default='gemini-2.5-pro')
    parser.add_argument('--output-dir', type=str, default='output-gemini')
    parser.add_argument('--num-processes', type=int, default=64, help="Number of processes to use")
    args = parser.parse_args()
    return args


def process_item(item):

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(base_url=GEMINI_BASE_URL, timeout=GEMINI_TIMEOUT)
    )

    cur_id = item['id']

    checklist = item['refined_question']
    clear_prompt = item['clear_prompt']
    task_type = 'understanding'
    category = item['category']
    ref_text = item['ref_text']
    ref_images = item['ref_image']
    input_images = item['input_image']
    output_text = item['output_text']

    if type(output_text) == list:
        output_text = output_text[0]

    os.makedirs(f'{total_save_dir}/{global_model_name}_split_results', exist_ok=True)

    output_json_file = f'{total_save_dir}/{global_model_name}_split_results/{cur_id}.json'

    if os.path.exists(output_json_file):
        with open(output_json_file) as f:
            exist_data = json.load(f)
        eval_successful = exist_data['eval_successful'] or 'eval_successful' not in exist_data
        if 'answer_list' in exist_data and 'refined_question' in exist_data:
            answer_list = exist_data['answer_list']
            refined_question = exist_data['refined_question']
            eval_successful = eval_successful and (len(answer_list) == len(refined_question))
        if not eval_successful:
            pass
        else:
            print(f'{cur_id} already exists, continue')
            log_dict = {}
            log_dict['id'] = cur_id
            log_dict['eval_successful'] = True
            log_dict['is_response_successful'] = True
            log_dict['is_parsed_successful'] = True
            log_dict['eval_model_name'] = global_model_name
            log_dict['reason'] = None
            return log_dict

    to_write_item = copy.deepcopy(item)
    pattern = r"(<image\d+>|<ref_images_placeholder_list>)"
    if "<image" in ref_text:
        ref_text = re.sub(r"<image(\d+)>", replace_image_tags, ref_text)

    if "<image" in output_text:
        output_text = re.sub(r"<image(\d+)>", replace_image_tags, output_text)

    parts = re.split(pattern,  eval_prompt.format(reference_text=ref_text, clear_prompt=clear_prompt, task_type=task_type, category=category, checklist=checklist, output_text=output_text))

    prompt_part_list = []
    for part in parts:
        if re.match(r'<image\d+>', part):
            prompt_part_list.append('<input_image>')
        elif re.match(r'<ref_images_placeholder_list>', part):
            prompt_part_list.append('<ref_images_placeholder_list>')
        elif re.match(r'<ref_image\d+>', part):
            prompt_part_list.append('<ref_image>')
        else:
            prompt_part_list.append(part)

    input_data = []
    input_image_id = 0
    ref_image_id = 0
    log_dict = {}

    try:
        for part in prompt_part_list:
            part = part.strip()
            if part == '<input_image>':
                image_p = input_images[input_image_id]
                full_image_path = os.path.join(image_dir, image_p)
                cur_image = Image.open(full_image_path)
                input_data.append(cur_image)
                input_image_id += 1

            elif part == '<ref_image>':
                image_p = input_images[ref_image_id]
                full_image_path = os.path.join(image_dir, image_p)
                cur_image = Image.open(full_image_path)
                input_data.append(cur_image)
                ref_image_id += 1

            elif part == '<ref_images_placeholder_list>':
                for image_p in ref_images:
                    full_image_path = os.path.join(image_dir, image_p)
                    cur_image = Image.open(full_image_path)
                    input_data.append(cur_image)
            else:
                if len(part) != 0 and part != ' ':
                    input_data.append(part)
    except:
        print(f'loading picture failed for {cur_id}')
        output_text = 'loading picture failed'
        to_write_item['output_text'] = output_text
        to_write_item['answer_list'] = output_text
        to_write_item['reason_list'] = output_text
        to_write_item['eval_model_name'] = global_model_name
        to_write_item['eval_successful'] = False
        to_write_item['is_response_successful'] = False
        to_write_item['is_parsed_successful'] = False

        log_dict = {}
        log_dict['id'] = cur_id
        log_dict['eval_successful'] = False
        log_dict['is_response_successful'] = False
        log_dict['is_parsed_successful'] = False
        log_dict['eval_model_name'] = global_model_name
        log_dict['reason'] = 'loading picture failed'
        return log_dict

    num_has_tries = 0
    is_response_successful = False
    output_text = ''

    while num_has_tries <= num_max_tries:
        try:
            response = client.models.generate_content(
                model=global_model_name,
                contents=input_data
            )
            output_text = response.text
            is_response_successful = True
            break
        except Exception as e:
            print(f'{cur_id} failed, retry for {e}')
            num_has_tries += 1
            continue

    if not is_response_successful:
        output_text = 'response failed'
        to_write_item['output_text'] = output_text
        to_write_item['answer_list'] = output_text
        to_write_item['reason_list'] = output_text
        to_write_item['eval_model_name'] = global_model_name
        to_write_item['eval_successful'] = False
        to_write_item['is_response_successful'] = False
        to_write_item['is_parsed_successful'] = False

        log_dict['id'] = cur_id
        log_dict['eval_successful'] = False
        log_dict['is_response_successful'] = False
        log_dict['is_parsed_successful'] = False
        log_dict['eval_model_name'] = global_model_name
        log_dict['reason'] = 'response failed'
        return log_dict


    output_text = output_text.replace('```python', '').replace('```json', '').replace('```', '').strip()
    is_parsed_successful = True
    try:
        output_dict = json.loads(output_text)
    except:
        try:
            output_dict = eval(output_text)
        except:
            output_dict = 'parsed_error'
            is_parsed_successful = False

    if type(output_dict) == dict:
        answer_list = output_dict['answer_list']
        reason_list = output_dict['reason_list']
    else:
        answer_list = 'parsed_error'
        reason_list = 'parsed_error'

    print(f'{answer_list=}, {reason_list=}')

    to_write_item['answer_list'] = answer_list
    to_write_item['reason_list'] = reason_list
    to_write_item['eval_model_name'] = global_model_name
    to_write_item['eval_successful'] = is_response_successful and is_parsed_successful
    to_write_item['is_response_successful'] = is_response_successful
    to_write_item['is_parsed_successful'] = is_parsed_successful
    to_write_item['output_eval_text'] = output_text

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(to_write_item, f, ensure_ascii=False, indent=4)

    
    log_dict = {}
    log_dict['id'] = cur_id
    log_dict['eval_successful'] = is_response_successful and is_parsed_successful
    log_dict['is_response_successful'] = is_response_successful
    log_dict['is_parsed_successful'] = is_parsed_successful
    log_dict['eval_model_name'] = global_model_name

    return log_dict



def eval_model(data, model_name, output_dir, num_processes):
    global total_save_dir
    total_save_dir = output_dir

    global global_model_name
    global_model_name = model_name

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_item, data), total=len(data)))

    output_log_file = f'{output_dir}/{model_name}_log.json'
    output_json_file = f'{output_dir}/{model_name}_results.json'

    results = sorted(results, key=lambda x: int(x['id']))

    with open(output_log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    failed_id = []
    for item in results:
        if not item['eval_successful']:
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


        src_json_file = f'{total_save_dir}/{model_name}_split_results/{cur_id}.json'
        if os.path.exists(src_json_file):
            with open(src_json_file, 'r', encoding='utf-8') as f:
                src_data = json.load(f)
            save_results.append(src_data)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=4)



def main():
    args = parse_args()
    model_name = args.model_name
    input_json = args.input_json
    output_dir = args.output_dir
    num_processes = args.num_processes
    sub_output_dir = f'{output_dir}/understanding_eval'

    os.makedirs(sub_output_dir, exist_ok=True)

    print(f'{model_name=}, {input_json=}, {output_dir=}')

    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        if item['task'] == 'understanding':
            data.append(item)
    
    eval_model(data, model_name, sub_output_dir, num_processes)




if __name__ == "__main__":
    main()
