import os
from util import *
import transformers
import pandas as pd
from tqdm import tqdm
from hallucination_editor import system_msg_eval


model_id = model_id_ls[-1]
model_id_format = model_id.split('/')[-1].replace('-', '_').lower()

folder_unfiltered = f"../data/questions/unfiltered/{model_id_format}"
folder_hallu = f"../data/questions/hallucination_all/{model_id_format}"

tok_qa = transformers.AutoTokenizer.from_pretrained(model_id)
model_qa = transformers.AutoModelForCausalLM.from_pretrained(model_id).to('cuda:0')

model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_eval = transformers.AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to('cuda:1')
tok_eval = transformers.AutoTokenizer.from_pretrained(model_id_eval)


def get_response(model, tok, messages, max_new_tokens=1): 
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


def evaluate_responses(model_eval, tok_eval, df, system_msg_eval, user_msg_eval_template="Text 1: {label} \nText 2: {output_qa}"):
    for i in df.index:
        label = df.loc[i, 'object']
        output_qa = df.loc[i, f"output_{model_id_format}"]
        eval_res = 0

        if output_qa.lower() in label.lower() or label.lower() in output_qa.lower() or 'unknown' in output_qa.lower():  # Rule-based fuzzy match
            eval_res = 1
            if output_qa.lower() == label.lower():
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Exact Match")
            else:
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Partial Match")
        else:
            user_msg_eval = user_msg_eval_template.format(label=label, output_qa=output_qa)
            messages_eval = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": user_msg_eval}]
            response_eval = get_response(model_eval, tok_eval, messages_eval)
            if response_eval != '0':
                print(f"Label: {label:<35} Prediction: {output_qa:<35} Evaluation: Semantic Match")
                eval_res = 1
                
        df.loc[i, f"eval_{model_id_format}"] = eval_res
    hallu_count = df[df[f'eval_{model_id_format}']==0].shape
    print(f"Hallucination ratio: {hallu_count[0]/len(df)} df_hallucination.shape: {hallu_count}")
    return df


for filename in os.listdir(folder_unfiltered):
    df = pd.read_csv(f"{folder_unfiltered}/{filename}")
    ls_output = []

    for i in tqdm(df.index, desc=f"Answering {filename}"):
        question = df.loc[i, 'question']
        if 'llama' in model_id_format.lower() or 'Mistral-7B-Instruct-v0.3' in model_id_format:
            messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
        elif 'gemma' in model_id_format.lower():
            messages_qa = [{"role": "user", "content": system_msg_qa+' '+question}]
        elif 'gpt' in model_id_format.lower():
            messages_qa = [f"{system_msg_qa} Question: {question} Answer:"] 
        
        output_qa = get_response(model_qa, tok_qa, messages_qa, max_new_tokens=16)
        ls_output.append(output_qa)
    
    df['topic'] = filename.replace('.csv', '')
    df[f"output_{model_id_format}"] = ls_output
    df[['topic', 'subject', 'relation', 'object', 'question', f'output_{model_id_format}']].to_csv(f"{folder_unfiltered}/{filename}", index=False)


system_msg_2 = "Given two texts, labeled as Text 1 and Text 2, output '1' if they have similar semantic meanings, are synonyms, \
or if one is a more specific or general version of the other; otherwise, output '0'. Do not repeat the question or provide any explanation."   

if not os.path.exists(folder_hallu):
    os.makedirs(folder_hallu)

for filename in os.listdir(folder_unfiltered):
    if os.path.exists(f"{folder_hallu}/{filename}"):
        continue
    df_qa = pd.read_csv(f"{folder_unfiltered}/{filename}")
    print(f"{filename}, df_qa.shape: {df_qa.shape}")
    print('Round 1.', end=' ')
    df_qa = evaluate_responses(model_eval, tok_eval, df_qa, system_msg_eval)
    df_hallu = df_qa[df_qa[f"eval_{model_id_format}"] == 0]

    print('Round 2.', end=' ')
    df_hallu = evaluate_responses(model_eval, tok_eval, df_hallu, system_msg_2)
    df_hallu = df_hallu[df_hallu[f"eval_{model_id_format}"] == 0]
    df_hallu.to_csv(f"{folder_hallu}/{filename}", index=False)