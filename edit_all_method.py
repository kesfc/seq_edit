import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams, WISEHyperParams, UltraEditHyperParams, AlphaEditHyperParams
import torch

if __name__ == "__main__":
    torch.cuda.empty_cache()
    question_type_ls = ['yes_questions', 'no_questions', 'locality_questions', 'rephrase_questions','multiple_choice_questions', 'reversed_relation_questions']
    
        # 'yes_questions', 'no_questions', 'locality_questions', 'rephrase_questions','multiple_choice_questions', 'reversed_relation_questions',
                        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3-8b')
    parser.add_argument('--data_size', default=None, type=int)
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results/sequential_edit_N', type=str)
    parser.add_argument('--edit_method', default=None, help='Edit method to use')
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=1, help='device of the local evaluation model')
    parser.add_argument('--dataset_dir', default='../data/questions/hallucination_final', type=str)
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default=None, help='model id of the local evaluation model')
    parser.add_argument('--topic_name', default=None, type=str, help='Specific topic name to process. If not provided, will process all topics.')
    parser.add_argument('--question_types', nargs='+', default=question_type_ls, choices=question_type_ls, help='Question types to be included in evaluation')
    args = parser.parse_args()
    start_time = time.time()

    topics= ['20topic_50each']
    for topic_name in topics:
        print(topic_name)
        editing_methods = ['GRACE', 'LoRA', 'FT-M', 'FT-L', 'ROME', 'WISE', 'ULTRA', 'MEMIT', 'ALPHA']
        if args.edit_method is not None:
            editing_methods = [args.edit_method]

        for editing_method in editing_methods[:]:
            if editing_method in ['FT-M', 'FT-L']:
                editing_hparams = FTHyperParams
            elif editing_method == 'ICL':
                editing_hparams = IKEHyperParams
            elif editing_method == 'ROME':
                editing_hparams = ROMEHyperParams
            elif editing_method == 'MEMIT':
                editing_hparams = MEMITHyperParams
            elif editing_method == 'LoRA':
                editing_hparams = LoRAHyperParams
            elif editing_method == 'GRACE':
                editing_hparams = GraceHyperParams
            elif editing_method == 'WISE':
                editing_hparams = WISEHyperParams
            elif editing_method == 'ULTRA':
                editing_hparams = UltraEditHyperParams
            elif editing_method == 'ALPHA':
                editing_hparams = AlphaEditHyperParams
            else:
                raise NotImplementedError

            hparams = editing_hparams.from_hparams(f'{args.hparams_dir}/{editing_method}/{args.model_name}')
            print(hparams)
            model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
            print(model_id_format)
            if os.path.exists(f"{args.dataset_dir}/{model_id_format}/{topic_name}.csv") == False:
                continue
            print(f'\nModel: {model_id_format}, Editing {topic_name} with {editing_method}...\n')
            if os.path.exists(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json'):
                print(f'Result {topic_name}_{editing_method}.json already exists\n')
                if args.overwrite_result:
                    print(f'Overwriting result {topic_name}_{editing_method}.json\n')
                else:
                    continue
            print(f"{args.dataset_dir}/{model_id_format}/{topic_name}.csv")

            df = pd.read_csv(f"{args.dataset_dir}/{model_id_format}/{topic_name}.csv")
            
            if args.data_size is not None:
                df = df[:args.data_size]
            targets = df['object'].tolist()
            subjects = df['subject'].tolist()
            questions = df['question'].tolist()
            paraphrased_questions = df['paraphrased_question'].tolist()
            locality_questions = {'locality': {'prompt': df['locality_question'].tolist()}}
            df['multiple_choice_full'] = df['question'] + ' ' + df['multiple_choice_with_letters']
            no_questions = {'no': {'prompt': df['no_question'].tolist(), 'ground_truth': ['No' for i in range(len(df))]}}
            yes_questions = {'yes': {'prompt': df['yes_question'].tolist(), 'ground_truth': ['Yes' for i in range(len(df))]}}
            reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist(), 'ground_truth': df['subject'].tolist()}}
            multiple_choice_questions = {'multiple_choice': {'prompt': df['multiple_choice_full'].tolist(), 'ground_truth': df['multiple_choice_labels'].tolist()}}
            print(f'Question types included in evaluation: {args.question_types}\n')

            hparams.device = args.device_edit  # overwrite device in hparams
            editor = BaseEditor.from_hparams(hparams)
            
            edit_kwargs = {
                'subject': subjects,
                'prompts': questions,
                'target_new': targets,
                'summary_metrics': True,
                'keep_original_weight': True,
                'eval_model_id': args.model_eval,
                'device_eval': f'cuda:{args.device_eval}',
            }
            
            if 'yes_questions' in args.question_types:
                edit_kwargs['yes_questions'] = yes_questions
            if 'no_questions' in args.question_types:
                edit_kwargs['no_questions'] = no_questions
            if 'locality_questions' in args.question_types:
                edit_kwargs['locality_inputs'] = locality_questions
            if 'rephrase_questions' in args.question_types:
                edit_kwargs['rephrase_prompts'] = paraphrased_questions
            if 'multiple_choice_questions' in args.question_types:
                edit_kwargs['multiple_choice_questions'] = multiple_choice_questions
            if 'reversed_relation_questions' in args.question_types:
                edit_kwargs['reversed_relation_questions'] = reversed_relation_questions

            metrics, edited_model, _ = editor.sequential_edit(**edit_kwargs)
            
            if not os.path.exists(f'{args.results_dir}/{model_id_format}'):
                os.makedirs(f'{args.results_dir}/{model_id_format}')
            json.dump(metrics, open(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json', 'w'), indent=4)
            
            print(f'\nModel: {model_id_format}, Editing {topic_name} with {editing_method} finished')
            del edited_model
            del editor
            gc.collect()
            torch.cuda.empty_cache()

        total_time = (time.time() - start_time) / 60 
        print(f'\nOverall running time for edit_all_method.py: {total_time:.2f} minutes')
        
    # Overall running time for edit_all_method.py: about 240 to 280 minutes