import json
import math
import torch
import random
import typing
import logging
import numpy as np
from tqdm import tqdm
from time import time
from typing import Optional, Union, List, Dict
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from easyeditor.util import nethook
from easyeditor.util.globals import *
from easyeditor.util.alg_dict import *
from easyeditor.models.melo.melo import LORA
from easyeditor.util.hparams import HyperParams
from easyeditor.editors.batch_editor import BatchEditor
from easyeditor.evaluate.evaluate_utils import test_generation_quality
from easyeditor.evaluate import compute_icl_edit_quality, compute_sent_metric


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
from openai import OpenAI

EVAL_MODEL = "gpt-4.1-nano" 
OPENAI_CLIENT = OpenAI()
LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

system_msg_eval = """You are a question grader. Given a question, ground truth answer, and model generated answer, you need to determine whether the model's answer is correct or not.
Your job is to determine whether the model generated answer is correct or not.
You should output 1 if the answer is correct, and 0 if the answer is incorrect.
"""
system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."
locality_msg_eval = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically; otherwise, output '0'. Do not repeat the question or provide any explanation."   


def make_logs():
    s_h = logging.StreamHandler()
    LOG.addHandler(s_h)


def get_all_acc_keys(dict_list):
    all_keys = set()
    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
    for dictionary in dict_list:
        recursive_keys(dictionary)
    return all_keys


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


def get_response(hparams, model, tok, messages, max_new_tokens=1, eval_flag=False, device_eval='cuda:0'):
    device = device_eval if eval_flag else hparams.device
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    if 'gpt' in hparams.model_name.lower() and eval_flag is False:
        msg_tokenized = tok(messages[0], return_tensors='pt').to(model.device)
    else:
        msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


def openai_eval(text1: str, text2: str, question: str) -> int:
    prompt = f"""Question: {question}\n Ground Truth: {text1}\n Model Output: {text2}
        Please output 1 if the model output is correct, and 0 if it is incorrect
        """
    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": system_msg_eval},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2,
        )
        out = resp.choices[0].message.content.strip()
        return 1 if out.startswith('1') else 0
    except Exception as e:
        LOG.warning(f"[openai_eval] Fallback to 0 due to error: {e}")
        return 0

def locailty_eval(text1, text2) -> int:
    prompt = f"""Text 1: {text1} \nText 2: {text2}"""
    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": locality_msg_eval},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2,
        )
        out = resp.choices[0].message.content.strip()
        return 1 if out.startswith('1') else 0
    except Exception as e:
        LOG.warning(f"[openai_eval] Fallback to 0 due to error: {e}")
        return 0


def evaluate_response(hparams, model_eval_unused, tok_eval_unused, prompt_qa, output_qa, label, device_eval_unused):
    if output_qa is None:
        output_qa = ""
        return 0, output_qa
    if label is None:
        return 0, output_qa
    else:
        response_eval = openai_eval(label, output_qa, prompt_qa)
    print(f"===== Question: {prompt_qa} | Prediction: {output_qa} | Label: {label} | Evaluation: {response_eval} =====")
    return int(response_eval), output_qa


def test_prediction_acc(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, pre_or_post, vanilla_generation=False, system_msg=system_msg_qa):
    if vanilla_generation and pre_or_post=='post':
        target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
        prompt_tok = tok_qa(prompt_qa, return_tensors="pt").to(model_qa.device)
        gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, 
                                      do_sample=False, temperature=0.0, use_cache=False)
        output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
        output_text = tok_qa.decode(output_text, skip_special_tokens=True)
        if label is None:
            return None, output_text
        return evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_text, label, device_eval)

    if isinstance(prompt_qa, list):
        for i, prompt in enumerate(prompt_qa):
            label_ = label[i] if label is not None else None
            return test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt, label_, system_msg)
    else:
        return test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, system_msg)


def test_prediction_acc_single(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, system_msg_qa):
    model_qa_name = hparams.model_name
    user_msg_qa = prompt_qa
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": user_msg_qa}]
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": system_msg_qa+' '+user_msg_qa}]
    elif 'vicuna' in model_qa_name.lower() or 'gpt' in model_qa_name.lower():
        messages_qa = [f"{system_msg_qa} Question: {user_msg_qa} Answer:"]
    else:
        messages_qa = [system_msg_qa+' '+user_msg_qa]

    output_qa = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)
    if label is None:
        return None, output_qa
    return evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_qa, label, device_eval)


def test_prediction_acc_multi_turn(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, multi_turn, yes_question, no_question, pre_or_post, vanilla_generation=False):
    acc_ls = []
    output_qa_ls = []

    if multi_turn == 'sure':
        questions_follow_up = [f"Are you sure about your answer to the original question? Respond with 'Yes' or 'No'." for _ in range(10)]
    elif multi_turn == 'yes':
        questions_follow_up = [f'Your answer to the original question is wrong. {yes_question} Respond with "Yes" or "No".' for _ in range(10)]
    label_follow_up = 'Yes'

    model_qa_name = hparams.model_name
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": prompt_qa}]
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": system_msg_qa+' '+prompt_qa}]
    else:
        messages_qa = [system_msg_qa+' '+prompt_qa]

    if vanilla_generation and pre_or_post=='post':
        target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
        prompt_tok = tok_qa(prompt_qa, return_tensors="pt").to(model_qa.device)
        gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, use_cache=False)
        output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
        current_output = tok_qa.decode(output_text, skip_special_tokens=True)
    else:
        current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)

    eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, prompt_qa, current_output, label, device_eval)
    acc_ls.append(eval_acc)
    output_qa_ls.append(current_output)

    for question in questions_follow_up:
        messages_qa.append({"role": "assistant", "content": current_output})
        messages_qa.append({"role": "user", "content": question})

        if vanilla_generation and pre_or_post=='post':
            target_new_tokens_len = len(tok_qa.encode(label, add_special_tokens=False)) if label is not None else 16
            formatted_input = tok_qa.apply_chat_template(messages_qa, tokenize=False, add_generation_prompt=True)
            prompt_tok = tok_qa(formatted_input, return_tensors="pt").to(model_qa.device)
            gen_token = model_qa.generate(**prompt_tok, max_new_tokens=target_new_tokens_len, pad_token_id=tok_qa.eos_token_id, use_cache=False)
            output_text = gen_token.detach().cpu().numpy().tolist()[0][-target_new_tokens_len:]
            current_output = tok_qa.decode(output_text, skip_special_tokens=True)
        else:
            current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)

        eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, question, current_output, label_follow_up, device_eval)
        acc_ls.append(eval_acc)
        output_qa_ls.append(current_output)

    return acc_ls, output_qa_ls


def compute_edit_or_rephrase_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval,
    tok_eval,
    device_eval,
    prompt: str,
    target_new: str,
    multi_turn: str,
    yes_question: str = None,
    no_question: str = None,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
    pre_or_post: str = 'pre'
) -> typing.Dict:
    key = 'rephrase' if test_rephrase else 'edit'
    if multi_turn is not None and key == 'edit':
        acc_ls, output_ls = test_prediction_acc_multi_turn(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, target_new, multi_turn,
                                                           yes_question, no_question, pre_or_post, vanilla_generation=hparams.alg_name=='GRACE')
        return {f"{key}_acc": [acc_ls[0]], f"{key}_output": [output_ls[0]], f"{key}_acc_multi_turn": acc_ls, f"{key}_output_multi_turn": output_ls}
    else:
        acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, target_new,
                                                pre_or_post, vanilla_generation=hparams.alg_name=="GRACE")
        return {f"{key}_acc": [acc], f"{key}_output": [model_output]}


def compute_general_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval,
    tok_eval,
    device_eval,
    question_key: str,
    prompt: typing.Union[str, List[str]],
    question_ground_truth: typing.Union[str, List[str]],
    pre_or_post: str,
) -> typing.Dict:
    acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt, question_ground_truth,
                                            pre_or_post, vanilla_generation=hparams.alg_name=='GRACE')
    return {f"{question_key}_acc": [acc], f"{question_key}_output": [model_output]}


def compute_multiple_choice_quality(hparams, model, tok, model_eval, tok_eval, device_eval, question_key, prompt_qa, label, pre_or_post):
    system_msg_multiple_choice = "Always respond to the multiple-choice question by selecting from the provided options. Only output the choice letter (A, B, C, or D)."
    acc, model_output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, prompt_qa, label,
                                            pre_or_post, vanilla_generation=hparams.alg_name=='GRACE', system_msg=system_msg_multiple_choice)
    return {f"{question_key}_acc": [acc], f"{question_key}_output": [model_output]}


def compute_edit_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval,
    tok_eval,
    device_eval,
    record: typing.Dict,
    multi_turn: str,
    eval_metric: str = 'token_em',
    test_generation = False,
    icl_pre_edit=True,
    pre_or_post='pre'
) -> typing.Dict:
    if isinstance(model, LORA):
        model=model.model

    target_new, ground_truth = (record[x] for x in ["target_new", "ground_truth"])
    edit_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if hparams.alg_name in ['ICL', 'IKE'] and icl_pre_edit == False:
        icl_prompt = f"New Fact: Q: {edit_prompts} A: {target_new}\n"
    else:
        icl_prompt = ""

    yes_question = record['yes_questions']['yes']['prompt'] if 'yes_questions' in record.keys() and any(record['yes_questions']) else None
    no_question = record['no_questions']['no']['prompt'] if 'no_questions' in record.keys() and any(record['no_questions']) else None

    ret = compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+edit_prompts, target_new,
                                           multi_turn, yes_question, no_question, eval_metric=eval_metric, pre_or_post=pre_or_post)

    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_questions'] = {}
    ret['no_questions'] = {}
    ret['multiple_choice_questions'] = {}
    ret['reversed_relation_questions'] = {}
    ret['questions_2hop'] = {}
    ret['questions_3hop'] = {}
    ret['questions_4hop'] = {}
    ret['questions_5hop'] = {}
    ret['questions_6hop'] = {}
    ret['harm_original_text'] = {}

    if rephrase_prompts is not None:
        ret.update(
            compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+rephrase_prompts, target_new,
                                             multi_turn, test_rephrase=True, eval_metric=eval_metric, pre_or_post=pre_or_post)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            locality_prompt = record['locality'][locality_key]['prompt']
            if isinstance(locality_prompt, list):
                locality_prompt = [e+icl_prompt for e in locality_prompt]
            else:
                locality_prompt = icl_prompt + locality_prompt
            ret['locality'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, locality_key, locality_prompt, None, pre_or_post=pre_or_post)
            )

    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            portability_prompt = record['portability'][portability_key]['prompt']
            if isinstance(portability_prompt, list):
                portability_prompt = [e+icl_prompt for e in portability_prompt]
            else:
                portability_prompt = icl_prompt + portability_prompt
            ret['portability'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, portability_key, portability_prompt, record['portability'][portability_key]['ground_truth'], pre_or_post)
            )

    if 'yes_questions' in record.keys() and any(record['yes_questions']):
        for key in record['yes_questions'].keys():
            yes_q = record['yes_questions'][key]['prompt']
            if isinstance(yes_q, list):
                yes_q = [e+icl_prompt for e in yes_q]
            else:
                yes_q = icl_prompt + yes_q
            ret['yes_questions'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, yes_q, record['yes_questions'][key]['ground_truth'], pre_or_post)
            )

    if 'no_questions' in record.keys() and any(record['no_questions']):
        for key in record['no_questions'].keys():
            no_q = record['no_questions'][key]['prompt']
            if isinstance(no_q, list):
                no_q = [e+icl_prompt for e in no_q]
            else:
                no_q = icl_prompt + no_q
            ret['no_questions'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, no_q, record['no_questions'][key]['ground_truth'], pre_or_post)
            )

    if 'multiple_choice_questions' in record.keys() and any(record['multiple_choice_questions']):
        for key in record['multiple_choice_questions'].keys():
            mcq = record['multiple_choice_questions'][key]['prompt']
            if isinstance(mcq, list):
                mcq = [e+icl_prompt for e in mcq]
            else:
                mcq = icl_prompt + mcq
            ret['multiple_choice_questions'].update(
                compute_multiple_choice_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, mcq, record['multiple_choice_questions'][key]['ground_truth'], pre_or_post)
            )

    if 'reversed_relation_questions' in record.keys() and any(record['reversed_relation_questions']):
        for key in record['reversed_relation_questions'].keys():
            rrq = record['reversed_relation_questions'][key]['prompt']
            if isinstance(rrq, list):
                rrq = [e+icl_prompt for e in rrq]
            else:
                rrq = icl_prompt + rrq
            ret['reversed_relation_questions'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, rrq, record['reversed_relation_questions'][key]['ground_truth'], pre_or_post)
            )

    if 'questions_2hop' in record.keys() and any(record['questions_2hop']):
        for key in record['questions_2hop'].keys():
            q = record['questions_2hop'][key]['prompt']
            if isinstance(q, list):
                q = [e+icl_prompt for e in q]
            else:
                q = icl_prompt + q
            ret['questions_2hop'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, q, record['questions_2hop'][key]['ground_truth'], pre_or_post)
            )

    if 'questions_3hop' in record.keys() and any(record['questions_3hop']):
        for key in record['questions_3hop'].keys():
            q = record['questions_3hop'][key]['prompt']
            if isinstance(q, list):
                q = [e+icl_prompt for e in q]
            else:
                q = icl_prompt + q
            ret['questions_3hop'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, q, record['questions_3hop'][key]['ground_truth'], pre_or_post)
            )

    if 'questions_4hop' in record.keys() and any(record['questions_4hop']):
        for key in record['questions_4hop'].keys():
            q = record['questions_4hop'][key]['prompt']
            if isinstance(q, list):
                q = [e+icl_prompt for e in q]
            else:
                q = icl_prompt + q
            ret['questions_4hop'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, q, record['questions_4hop'][key]['ground_truth'], pre_or_post)
            )

    if 'questions_5hop' in record.keys() and any(record['questions_5hop']):
        for key in record['questions_5hop'].keys():
            q = record['questions_5hop'][key]['prompt']
            if isinstance(q, list):
                q = [e+icl_prompt for e in q]
            else:
                q = icl_prompt + q
            ret['questions_5hop'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, q, record['questions_5hop'][key]['ground_truth'], pre_or_post)
            )

    if 'questions_6hop' in record.keys() and any(record['questions_6hop']):
        for key in record['questions_6hop'].keys():
            q = record['questions_6hop'][key]['prompt']
            if isinstance(q, list):
                q = [e+icl_prompt for e in q]
            else:
                q = icl_prompt + q
            ret['questions_6hop'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, q, record['questions_6hop'][key]['ground_truth'], pre_or_post)
            )

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model, tok=tok, prefixes=edit_prompts if isinstance(edit_prompts,list) else [edit_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_edit_quality_sequential_icl(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    model_eval, 
    tok_eval, 
    device_eval,
    record: typing.Dict,                
    records: typing.List[typing.Dict],   
    multi_turn: str,
    eval_metric: str = 'token_em',
    test_generation: bool = False,
    icl_pre_edit: bool = True,
    pre_or_post: str = 'pre'
) -> typing.Dict:

    if isinstance(model, LORA):
        model = model.model

    target_new, ground_truth = (record[x] for x in ["target_new", "ground_truth"])
    edit_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record else None

    icl_prompt = ""
    if hparams.alg_name in ['ICL', 'IKE'] and icl_pre_edit is False:
        for r in records:
            icl_prompt += f"New Fact: Q: {r['prompt']} A: {r['target_new']}\n"

    
    yes_question = record['yes_questions']['yes']['prompt'] if 'yes_questions' in record.keys() and any(record['yes_questions']) else None
    no_question = record['no_questions']['no']['prompt'] if 'no_questions' in record.keys() and any(record['no_questions']) else None
    ret = compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+edit_prompts, target_new, 
                                           multi_turn, yes_question, no_question, eval_metric=eval_metric, pre_or_post=pre_or_post)

    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_questions'] = {}
    ret['no_questions'] = {}
    ret['multiple_choice_questions'] = {}
    ret['reversed_relation_questions'] = {}
    ret['questions_2hop'] = {}
    ret['questions_3hop'] = {}
    ret['questions_4hop'] = {}
    ret['questions_5hop'] = {}
    ret['questions_6hop'] = {}
    ret['harm_original_text'] = {}

    if rephrase_prompts is not None:
        ret.update(
            compute_edit_or_rephrase_quality(hparams, model, tok, model_eval, tok_eval, device_eval, icl_prompt+rephrase_prompts, target_new, 
                                             multi_turn, test_rephrase=True, eval_metric=eval_metric, pre_or_post=pre_or_post)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            locality_prompt = record['locality'][locality_key]['prompt']
            if isinstance(locality_prompt, list):
                locality_prompt = [e+icl_prompt for e in locality_prompt]
            else:
                locality_prompt = icl_prompt + locality_prompt
            ret['locality'].update(
                compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, locality_key, locality_prompt, None, pre_or_post=pre_or_post)  # record['locality'][locality_key]['ground_truth'] ground_truth is not used in locality evaluation
            )

    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            portability_prompt = record['portability'][portability_key]['prompt']
            if isinstance(portability_prompt, list):
                portability_prompt = [e+icl_prompt for e in portability_prompt]
            else:
                portability_prompt = icl_prompt + portability_prompt
            ret['portability'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, portability_key, portability_prompt, record['portability'][portability_key]['ground_truth'], pre_or_post))
    
    if 'yes_questions' in record.keys() and any(record['yes_questions']):
        for key in record['yes_questions'].keys():
            yes_question = record['yes_questions'][key]['prompt']
            if isinstance(yes_question, list):
                yes_question = [e+icl_prompt for e in yes_question]
            else:
                yes_question = icl_prompt + yes_question
            ret['yes_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, yes_question, record['yes_questions'][key]['ground_truth'], pre_or_post))

    if 'no_questions' in record.keys() and any(record['no_questions']):
        for key in record['no_questions'].keys():
            no_question = record['no_questions'][key]['prompt']
            if isinstance(no_question, list):
                no_question = [e+icl_prompt for e in no_question]
            else:
                no_question = icl_prompt + no_question
            ret['no_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, no_question, record['no_questions'][key]['ground_truth'], pre_or_post))

    if 'multiple_choice_questions' in record.keys() and any(record['multiple_choice_questions']):
        for key in record['multiple_choice_questions'].keys():
            multiple_choice_question = record['multiple_choice_questions'][key]['prompt']
            if isinstance(multiple_choice_question, list):
                multiple_choice_question = [e+icl_prompt for e in multiple_choice_question]
            else:
                multiple_choice_question = icl_prompt + multiple_choice_question
            ret['multiple_choice_questions'].update(compute_multiple_choice_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, multiple_choice_question, record['multiple_choice_questions'][key]['ground_truth'], pre_or_post))

    if 'reversed_relation_questions' in record.keys() and any(record['reversed_relation_questions']):
        for key in record['reversed_relation_questions'].keys():
            reversed_relation_question = record['reversed_relation_questions'][key]['prompt']
            if isinstance(reversed_relation_question, list):
                reversed_relation_question = [e+icl_prompt for e in reversed_relation_question]
            else:
                reversed_relation_question = icl_prompt + reversed_relation_question
            ret['reversed_relation_questions'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, reversed_relation_question, record['reversed_relation_questions'][key]['ground_truth'], pre_or_post))

    if 'questions_2hop' in record.keys() and any(record['questions_2hop']):
        for key in record['questions_2hop'].keys():
            question = record['questions_2hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_2hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_2hop'][key]['ground_truth'], pre_or_post))

    if 'questions_3hop' in record.keys() and any(record['questions_3hop']):
        for key in record['questions_3hop'].keys():
            question = record['questions_3hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_3hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_3hop'][key]['ground_truth'], pre_or_post))

    if 'questions_4hop' in record.keys() and any(record['questions_4hop']):
        for key in record['questions_4hop'].keys():
            question = record['questions_4hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_4hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_4hop'][key]['ground_truth'], pre_or_post))

    if 'questions_5hop' in record.keys() and any(record['questions_5hop']):
        for key in record['questions_5hop'].keys():
            question = record['questions_5hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question

            ret['questions_5hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_5hop'][key]['ground_truth'], pre_or_post))

    if 'questions_6hop' in record.keys() and any(record['questions_6hop']):
        for key in record['questions_6hop'].keys():
            question = record['questions_6hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question
            ret['questions_6hop'].update(compute_general_quality(hparams, model, tok, model_eval, tok_eval, device_eval, key, question, record['questions_6hop'][key]['ground_truth'], pre_or_post)) 

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=edit_prompts if isinstance(edit_prompts,list) else [edit_prompts,], max_out_len=100, vanilla_generation=False)
    return ret


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif self.model_name in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3'] and hparams.alg_name == 'ROME':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower() or 'vicuna' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                print("WARNING: Probably Not Implemented")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams


    

    def sequential_edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             yes_questions: Optional[Dict] = None,
             no_questions: Optional[Dict] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             multiple_choice_questions: Optional[Dict] = None,
             reversed_relation_questions: Optional[Dict] = None,
             harm_original_text: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False,
             eval_model_id='ignored',
             device_eval='ignored',
             multi_turn=None,
             **kwargs
             ):
    
        test_generation = kwargs.get('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else:
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if "requests" in kwargs:
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(
                prompts, target_new, ground_truth,
                rephrase_prompts, yes_questions, no_questions,
                locality_inputs, portability_inputs,
                multiple_choice_questions, reversed_relation_questions,
                harm_original_text, **kwargs
            )

        #pre eval
        all_metrics = []
        for i, request in enumerate(tqdm(requests, desc="Pre-eval (original model)")):
            pre_metrics = compute_edit_quality(
                self.hparams, self.model, self.tok,
                model_eval=None, tok_eval=None, device_eval=None,
                record=request, multi_turn=multi_turn,
                test_generation=test_generation, pre_or_post='pre'
            )
            all_metrics.append({"pre": pre_metrics})

        #edit
        edited_model = self.model
        per_edit_exec_times = []
        if self.alg_name in ['GRACE', 'LoRA', 'FT-M', 'FT-L', 'ROME', 'WISE', 'ULTRAEDIT', 'ALPHAEDIT', 'MEMIT']:
            print("editing")
            edited_model, weights_copy = self.apply_algo(
                    edited_model,
                    self.tok,
                    requests,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
        if isinstance(edited_model, LORA):
            edited_model = edited_model.model

        # post eval
        for i, request in enumerate(tqdm(requests, desc="Post-eval (final model)")):
            if self.alg_name in ['IKE', 'ICL']:
                post_metrics = compute_edit_quality_sequential_icl(
                    self.hparams, edited_model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                    record=request, records=requests, multi_turn=multi_turn,
                    test_generation=test_generation, icl_pre_edit = False, pre_or_post='post'
                )
            else:
                post_metrics = compute_edit_quality(
                    self.hparams, edited_model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                    record=request, multi_turn=multi_turn,
                    test_generation=test_generation, pre_or_post='post'
                )
            all_metrics[i]["post"] = post_metrics

            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for pre_edit_output, post_edit_output in zip(
                        all_metrics[i]['pre']['locality'][f'{locality_key}_output'],
                        all_metrics[i]['post']['locality'][f'{locality_key}_output']
                    ):
                        acc = locailty_eval(pre_edit_output, post_edit_output)
                        locality_result.append(acc)
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['pre']['locality'].pop(f'{locality_key}_acc', None)

        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            mean_metrics = dict()
            for eval_key in ["pre", "post"]:
                mean_metrics[eval_key] = dict()
                for key in ["edit_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval_key].keys():
                        mean_metrics[eval_key][key] = np.mean([m[eval_key][key] for m in all_metrics])
                for key in ["locality", "portability", "yes_questions", "no_questions", "multiple_choice_questions", "reversed_relation_questions"]:
                    if key in all_metrics[0][eval_key].keys() and all_metrics[0][eval_key][key] != {}:
                        mean_metrics[eval_key][key] = dict()
                        for lkey in get_all_acc_keys(all_metrics):
                            vals = [m[eval_key][key][lkey] for m in all_metrics if lkey in m[eval_key][key].keys()]
                            if len(vals) > 0:
                                mean_metrics[eval_key][key][lkey] = np.mean(vals)

        return all_metrics, edited_model, {}
    def edit(self,
            prompts: Union[str, List[str]],
            target_new: Union[str, List[str]],
            ground_truth: Optional[Union[str, List[str]]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            yes_questions: Optional[Dict] = None,
            no_questions: Optional[Dict] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            multiple_choice_questions: Optional[Dict] = None,
            reversed_relation_questions: Optional[Dict] = None,
            harm_original_text: Optional[Union[str, List[str]]] = None,
            keep_original_weight=False,
            verbose=True,
            summary_metrics=False,
            eval_model_id='ignored',
            device_eval='ignored',
            multi_turn=None,
            **kwargs
            ):
        test_generation = kwargs.get('test_generation', False)
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):
            self.hparams.batch_size = 1
            
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else:
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if "requests" in kwargs:
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(
                prompts, target_new, ground_truth,
                rephrase_prompts, yes_questions, no_questions,
                locality_inputs, portability_inputs,
                multiple_choice_questions, reversed_relation_questions,
                harm_original_text, **kwargs
            )

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, print('Single Edit, pls set the batch_size to 1....')

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for _ in requests:
                all_metrics.append({"pre": {}})

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": {}
                })
                if verbose:
                    LOG.info(f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}")
            return all_metrics, edited_model, weights_copy

        all_metrics = []

        for i, request in enumerate(tqdm(requests, desc="Pre-eval (original model)")):
            print(request)
            if self.alg_name in ['IKE', 'ICL']:
                pre_metrics = compute_edit_quality(
                    self.hparams, self.model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                record=request, multi_turn=multi_turn,
                    test_generation=test_generation, icl_pre_edit=True, pre_or_post='pre'
                )
            else:
                pre_metrics = compute_edit_quality(
                    self.hparams, self.model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                record=request, multi_turn=multi_turn,
                    test_generation=test_generation, pre_or_post='pre'
                )
            all_metrics.append({"pre": pre_metrics})

        for i, request in enumerate(tqdm(requests, desc="Editing (single per request)")):
            print(request)
            start = time()
            if self.alg_name in ['IKE', 'ICL']:
                edited_model, weights_copy = self.model, {}
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            if self.alg_name in ['IKE', 'ICL']:
                post_metrics = compute_edit_quality(
                    self.hparams, edited_model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                record=request, multi_turn=multi_turn,
                    test_generation=test_generation, icl_pre_edit=False, pre_or_post='post'
                )
            else:
                post_metrics = compute_edit_quality(
                    self.hparams, edited_model, self.tok,
                    model_eval=None, tok_eval=None, device_eval=None,
                record=request, multi_turn=multi_turn,
                    test_generation=test_generation, pre_or_post='post'
                )

            all_metrics[i].update({
                'case_id': i,
                "requested_edit": request,
                "time": exec_time,
                "post": post_metrics,
            })

            if "metric_kwargs" in kwargs:
                all_metrics[i].update(compute_sent_metric(
                    self.model, edited_model, self.model_name, self.hparams, self.tok,
                    metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device
                ))
            if self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                if hasattr(self.model, "peft_config"):
                    del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' and not keep_original_weight:
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for question, pre_edit_output, post_edit_output in zip(
                        all_metrics[i]['requested_edit']['locality']['locality']['prompt'],
                        all_metrics[i]['pre']['locality'][f'{locality_key}_output'],
                        all_metrics[i]['post']['locality'][f'{locality_key}_output']
                    ):
                        acc, _ = evaluate_response(self.hparams, None, None, question, pre_edit_output, post_edit_output, None)
                        locality_result.append(acc)
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['pre']['locality'].pop(f'{locality_key}_acc', None)

            if verbose:
                LOG.info(f"{i} editing: {request['prompt']} -> {request['target_new']}")


        if isinstance(edited_model, LORA):
            edited_model = edited_model.model

        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            mean_metrics = dict()
            for eval_key in ["pre", "post"]:
                mean_metrics[eval_key] = dict()
                for key in ["edit_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval_key].keys():
                        mean_metrics[eval_key][key] = np.mean([m[eval_key][key] for m in all_metrics])
                for key in ["locality", "portability", "yes_questions", "no_questions", "multiple_choice_questions", "reversed_relation_questions"]:
                    if key in all_metrics[0][eval_key].keys() and all_metrics[0][eval_key][key] != {}:
                        mean_metrics[eval_key][key] = dict()
                        for lkey in get_all_acc_keys(all_metrics):
                            vals = [m[eval_key][key][lkey] for m in all_metrics if lkey in m[eval_key][key].keys()]
                            if len(vals) > 0:
                                mean_metrics[eval_key][key][lkey] = np.mean(vals)
            mean_metrics["time"] = np.mean([m["time"] for m in all_metrics])
            print("Metrics Summary: ", mean_metrics)
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          yes_questions: Optional[Dict] = None,
                          no_questions: Optional[Dict] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          multiple_choice_questions: Optional[Dict] = None,
                          reversed_relation_questions: Optional[Dict] = None,
                          harm_original_text: Union[str, List[str]] = None,
                          **kwargs
                          ):
        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {},
            'yes_questions': {},
            'no_questions': {},
            'multiple_choice_questions': {},
            'reversed_relation_questions': {},
            'harm_original_text': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')
            for i, request in enumerate(requests):
                request.update({'subject': kwargs['subject'][i]})

        if harm_original_text is not None:
            if isinstance(harm_original_text, str):
                harm_original_text = [harm_original_text,]
            for i, request in enumerate(requests):
                request.update({'harm_original_text': harm_original_text[i]})

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]
            for i, request in enumerate(requests):
                request.update({'rephrase_prompt': rephrase_prompts[i]})

        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {locality_key: {'prompt': locality_inputs[locality_key]['prompt'][i]}}
                        )

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) == len(requests), print('One Edit instance needs one portability input.....')
                for i, request in enumerate(requests):
                    if portability_inputs[portability_key]['prompt'][i] is not None:
                        request['portability'].update(
                            {portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }}
                        )

        if yes_questions is not None:
            for key in yes_questions.keys():
                if isinstance(yes_questions[key]['prompt'], str):
                    yes_questions[key]['prompt'] = [yes_questions[key]['prompt'],]
                    yes_questions[key]['ground_truth'] = [yes_questions[key]['ground_truth'], ]
                assert len(yes_questions[key]['prompt']) == len(yes_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')
                for i, request in enumerate(requests):
                    if yes_questions[key]['prompt'][i] is not None:
                        request['yes_questions'].update({key: {'prompt': yes_questions[key]['prompt'][i], 'ground_truth': yes_questions[key]['ground_truth'][i]}})

        if no_questions is not None:
            for key in no_questions.keys():
                if isinstance(no_questions[key]['prompt'], str):
                    no_questions[key]['prompt'] = [no_questions[key]['prompt'],]
                    no_questions[key]['ground_truth'] = [no_questions[key]['ground_truth'], ]
                assert len(no_questions[key]['prompt']) == len(no_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')
                for i, request in enumerate(requests):
                    if no_questions[key]['prompt'][i] is not None:
                        request['no_questions'].update({key: {'prompt': no_questions[key]['prompt'][i],  'ground_truth': no_questions[key]['ground_truth'][i]}})

        if multiple_choice_questions is not None:
            for key in multiple_choice_questions.keys():
                if isinstance(multiple_choice_questions[key]['prompt'], str):
                    multiple_choice_questions[key]['prompt'] = [multiple_choice_questions[key]['prompt'],]
                    multiple_choice_questions[key]['ground_truth'] = [multiple_choice_questions[key]['ground_truth'], ]
                assert len(multiple_choice_questions[key]['prompt']) == len(multiple_choice_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')
                for i, request in enumerate(requests):
                    if multiple_choice_questions[key]['prompt'][i] is not None:
                        request['multiple_choice_questions'].update({key: {'prompt': multiple_choice_questions[key]['prompt'][i], 'ground_truth': multiple_choice_questions[key]['ground_truth'][i]}})

        if reversed_relation_questions is not None:
            for key in reversed_relation_questions.keys():
                if isinstance(reversed_relation_questions[key]['prompt'], str):
                    reversed_relation_questions[key]['prompt'] = [reversed_relation_questions[key]['prompt'],]
                    reversed_relation_questions[key]['ground_truth'] = [reversed_relation_questions[key]['ground_truth'], ]
                assert len(reversed_relation_questions[key]['prompt']) == len(reversed_relation_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')
                for i, request in enumerate(requests):
                    if reversed_relation_questions[key]['prompt'][i] is not None:
                        request['reversed_relation_questions'].update({key: {'prompt': reversed_relation_questions[key]['prompt'][i], 'ground_truth': reversed_relation_questions[key]['ground_truth'][i]}})

        return requests


    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        keep_original_weight=False,
        epoch: int=5,
    ):
        """
        Batch edit for batchable methods.
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')

        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=keep_original_weight,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy
