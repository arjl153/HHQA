import json
import time
import random
import openai
import argparse
from tqdm import tqdm

def exp_backoff_retry(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (openai.error.APIError,),
):
 
    def wrapper(*args, **kwargs):
        cur_retries, cur_delay = 0, initial_delay
 
        while True:
            try:
                return func(*args, **kwargs)
 
            except Exception as e:
                cur_retries += 1
                print(e) 
                if cur_retries > max_retries:
                    raise Exception(
                        f"Your maximum number of retries ({max_retries}) have exceeded."
                    ) 
                cur_delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(cur_delay)
 
    return wrapper
    
@exp_backoff_retry
def generate_response(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def form_prompt(sample, resource):
    question_resource = resource[sample['question_id']]

    prompt_list = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '''मुझे अब आपको एक प्रश्न-उत्तर रोबोट के उत्पादन की जाँच करने की आवश्यकता है कि क्या उसमें कोई 'भ्रम' है। मैं आपको कुछ सही उत्तरों के उदाहरण प्रदान करूंगा, और निर्णय लेने के मानदंड निम्नलिखित हैं:
    1. आपको पहले यह निर्धारित करना होगा कि प्रश्न-उत्तर रोबोट का उत्तर सुचारू है या नहीं। यदि उत्तर में बहुत सारे गड़बड़ी है, जैसे कि बहुत सारे गड़बड़ वाक्य हैं, तो इसे 'भ्रम' माना जाना चाहिए।
    2. दूसरे, आपको यह निर्धारित करना होगा कि क्या प्रश्न-उत्तर रोबोट ने सवाल का सीधा जवाब दिया है। यदि प्रश्न-उत्तर रोबोट का उत्तर सही जानकारी से भरा हुआ है लेकिन सीधे सवाल का जवाब नहीं देता है, तो इसे भी 'भ्रम' माना जाना चाहिए।
    3. यदि प्रश्न-उत्तर रोबोट का उत्तर सही उत्तर उदाहरणों से निकाला नहीं जा सकता है, या यदि इसमें सही उत्तर उदाहरणों के विपरीत जानकारी होती है, तो इसे 'भ्रम' माना जाना चाहिए।
    4. यदि प्रश्न-उत्तर रोबोट का उत्तर किसी भी सही उत्तर उदाहरण द्वारा समर्थित किया जा सकता है, तो इसे 'भ्रम' नहीं माना जाना चाहिए।
    5. अगर प्रश्न-उत्तर रोबोट का उत्तर सही उत्तर उदाहरणों द्वारा सीधे समर्थित नहीं किया जा सकता है, तो आपको यह तय करना होगा कि उत्तर का अर्थ सही उत्तर उदाहरणों के समान है या नहीं। अगर उत्तर का अर्थ सही उत्तर उदाहरणों के समान होता है, तो इसे 'भ्रम' के रूप में नहीं माना जाना चाहिए।
    6. अगर सही उत्तर उदाहरण में 'इस प्रश्न का उत्तर नहीं दिया जा सकता' जैसी बात होती है, तो जब प्रश्न-उत्तर रोबोट 'मुझे नहीं पता' जैसा उत्तर देता है, तो इसे 'भ्रम' के रूप में नहीं माना जाना चाहिए। '''
        }
    ]
    
    prompt_list.append({'role': 'assistant', 'content': 'समझ गया, मैं आपके द्वारा प्रदान किए गए उदाहरणों और मानदंडों के आधार पर यह निर्णय लूंगा कि प्रश्न-उत्तर रोबोट के उत्पादन में भ्रम हैं या नहीं। कृपया उस प्रश्न को प्रदान करें जिसका निर्णय किया जाना है, सही और गलत उत्तरों के उदाहरण, और प्रश्न-उत्तर रोबोट का उत्तर।'})
    prompt_list.append({'role': 'user', 'content': ''})

    user_prompt = 'प्रश्न：{}\n\n'.format(question_resource['Question'].strip())
    user_prompt += 'सही उत्तरों के उदाहरण इस प्रकार हैं:\n'

    if 'Best Answer1' not in question_resource:
        user_prompt += '1. {}\n'.format(question_resource['Best Answer'].strip())
        sample['Best_Answer'] = question_resource['Best Answer'].strip()

    else:
        cnt = 1
        for i in range(1,5):
            if 'Best Answer{}'.format(str(i)) in question_resource:
                correct_answer_key = 'Best Answer{}'.format(str(i))
                if question_resource[correct_answer_key] != '':
                    user_prompt += '{}. {}\n'.format(str(cnt), question_resource[correct_answer_key].strip())
                    sample['Best_Answer{}'.format(str(i))] = question_resource[correct_answer_key].strip()
                    cnt += 1

    user_prompt += '\nप्रश्न-उत्तर रोबोट का उत्तर इस प्रकार है：\n'
    user_prompt += '{}\n\n'.format(sample['response'].strip())
    user_prompt += "अब, कृपया यह निर्णय लें कि प्रश्न-उत्तर रोबोट के उत्तर में 'भ्रम' हैं या नहीं। केवल 'हाँ' या 'नहीं' के रूप में उत्तर दें।"

    prompt_list[-1]['content'] = user_prompt

    return sample, prompt_list

def evaluate(args, resource):
    with open(args.response_file_name, 'r') as f:
        response_data = json.load(f)

    final_outputs = []
    no_of_correct = 0

    for row in tqdm(response_data):
        sample, prompt_list = form_prompt(row, resource)
        max_try, try_count = 5, 0
        inv_jdg = False
        while True:
            try_count += 1
            responses = generate_response(
                model="gpt-3.5-turbo",
                messages=prompt_list,
                temperature=0,
                top_p=0.5,
                n=5,
                max_tokens=10,
            )
            flg = True
            for choice in responses['choices']:
                if choice['message']['content'] != 'हाँ' and choice['message']['content'] != 'नहीं':
                    flg = False
                    break
            if flg:
                break
            if try_count >= max_try:
                inv_jdg = True
                break
            time.sleep(1)
        time.sleep(2)

        if inv_jdg is True:
            sample['is_hallucination'] = "Invalid Answer"
            final_outputs.append(sample)

        else:
            outputs = []
            for choice in responses['choices']:
                outputs.append(choice['message']['content'])
            
            if outputs.count('Yes') > 2:
                sample['is_hallucination'] = True
            else:
                sample['is_hallucination'] = False
                if sample['response'] != '':
                    no_of_correct += 1
                else:
                    sample['is_hallucination'] = True
            final_outputs.append(sample)
        break

    with open(args.result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_outputs, f, indent=2, ensure_ascii=False)
            
    with open(args.metric_save_path, 'w', encoding='utf-8') as f:
        f.write('Non hallucination rate: {:.2f}%'.format(no_of_correct*100/len(response_data)))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file_name', type=str, required=True)
    parser.add_argument('--result_save_path', type=str, default='results.json')
    parser.add_argument('--metric_save_path', type=str, default='non_hallucination_rate.txt')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--organization', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    openai.api_key = args.api_key
    openai.organization = args.organization

    with open('HindiHalluQA_2.json', 'r') as f:
        resource = {itm['question_id']: itm for itm in json.loads(f.read())}

    print('Caculating hallucination for {}---'.format(args.response_file_name))
    evaluate(args, resource)