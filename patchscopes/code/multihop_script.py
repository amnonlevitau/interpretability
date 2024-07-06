# %% [markdown]
# # Imports

# %%
from ast import literal_eval
import functools
import json
import os
import random
import shutil
import json 

# Scienfitic packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import datasets
from torch import cuda

torch.set_grad_enabled(False)

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns


print("Using GPU:", torch.cuda.get_device_name(device=None))
os.makedirs('/vol/scratch/jonathany/.cache', exist_ok=True)
os.environ['HF_HOME'] = '/vol/scratch/jonathany/.cache'
print("Current directory ", os.path.dirname(os.path.realpath(__file__)))


sns.set(context="notebook",
        rc={"font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

# Utilities

from general_utils import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_from_input,
)

from patchscopes_utils import *

import tqdm

tqdm.tqdm.pandas()

# %%
batch_size_scale = 8
# Load model

# model_name = "lmsys/vicuna-7b-v1.1"
model_name = "lmsys/vicuna-13b-v1.1"

print(f"Using model {model_name}")
torch_dtype = torch.float16
# model_name = "meta-llama/Meta-Llama-3-8B"
# torch_dtype = torch.bfloat16

sos_tok = False


my_device = torch.device("cuda:0")

mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch_dtype,
    device=my_device,
)
model_name = os.path.basename(model_name)
mt.set_hs_patch_hooks = set_hs_patch_hooks_llama_batch
mt.model.eval()

def count_tokens(s):
    inp = make_inputs(mt.tokenizer, [s], 'cpu') 
    return len(inp["input_ids"][0])
# %% [markdown]
# # MultiHop reasoning experiments

# %%
def generate_baseline_multihop(
        mt, df, batch_size=256 // batch_size_scale, max_gen_len=10, cases_list=None
):
    def _generate_baseline_single_batch(batch_df, cases_list_inner):
        batch_size = len(batch_df)
        if cases_list_inner is None:
            cases = [("baseline_hop2", "hop2"),
                    ("baseline_hop3", "hop3"),
                    ("baseline_multihop3", "hop3"),
                    ]
        else:
            cases = cases_list_inner
        results = {}
        for target_col, object_col in cases:
            target_baseline_batch = np.array(batch_df[target_col])
            object_batch = np.array(batch_df[object_col])

            # Step 0: run the the model on target prompt baseline (having the subject token in input rather than patched)
            # The goal of this step is to calculate whether the model works correctly by default, and to calculate surprisal
            inp_target_baseline = make_inputs(mt.tokenizer, target_baseline_batch, mt.device)
            seq_len_target_baseline = len(inp_target_baseline["input_ids"][0])
            output_target_baseline_toks = mt.model.generate(
                inp_target_baseline["input_ids"],
                max_length=seq_len_target_baseline + max_gen_len,
                pad_token_id=mt.model.generation_config.eos_token_id,
            )[:, seq_len_target_baseline:]
            generations_baseline = decode_tokens(mt.tokenizer, output_target_baseline_toks)
            generations_baseline_txt = np.array([" ".join(sample_gen) for sample_gen in generations_baseline])

            is_correct_baseline = np.array([
                (object_batch[i] in generations_baseline_txt[i] or
                 object_batch[i].replace(" ", "") in generations_baseline_txt[i].replace(" ", ""))
                for i in range(batch_size)
            ])
            results.update(
                {
                    f"generations_{target_col}": [i.replace('\n', ' \\n ') for i in generations_baseline_txt],
                    f"is_correct_{target_col}": is_correct_baseline,
                }
            )

        return results

    results = {}
    n_batches = len(df) // batch_size
    if len(df) % batch_size != 0:
        n_batches += 1
    for i in tqdm.tqdm(range(n_batches)):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_results = _generate_baseline_single_batch(cur_df, cases_list_inner=cases_list)
        for key, value in batch_results.items():
            if key in results:
                results[key] = np.concatenate((results[key], value))
            else:
                results[key] = value

    return results


# %% [markdown]
# # Experiment 1: Multihop Product Company CEO tuples
# 
# This is a subset made only from (product, company) and (company, CEO) tuples from the LRE dataset.
# We only picked 3 (company, CEO) tuples, and 15 (product, company) tuples for each that the model is more likely to know the answer to.
# 
# This is an exploratory experiment. There is a more complete experiment later in the colab.
# Hop 1: Product
# Hop 2: company
# Hop 3: CEO

# %%
multihop_samples = {
    ("Satya Nadella", "Microsoft"): ["WinDbg", ".NET Framework", "Internet Explorer", "MS-DOS", "Office Open XML",
                                     "TypeScript", "Bing Maps Platform", "Outlook Express", "PowerShell", "Windows 95",
                                     "Xbox 360", "Zune", "Visual Basic Script", "Virtual Hard Disk", "Robocopy",
                                     ],
    ("Tim Cook", "Apple"): ["Siri", "App Store", "CarPlay", "MacBook Air", "Xcode",
                            "macOS", "iWork", "Safari", "QuickTime", "TextEdit",
                            "WebKit", "QuickDraw", "Time Machine (macOS)", "MessagePad", "Macbook Pro",
                            ],
    ("Sundar Pichai", "Google"): ["Chromecast", "Chromebook", "Wear OS", "G Suite", "Picasa",
                                  "WebP Lossless", "General Transit Feed Specification Lossless", "Cloud Spanner",
                                  "Android TV", "Android Runtime",
                                  "Android Jelly Bean", "Android Auto", "App Inventor", "Chromebook Pixel",
                                  "Project Ara",
                                  ]
}


def generate_multihop_data_ceo(fdir_out="./outputs/factual", batch_size=512 // batch_size_scale, max_gen_len=20,
                               replace=False):
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    fname_out = "multihop_product_company_ceo"
    if not replace and os.path.exists(f"{fdir_out}/{fname_out}.pkl"):
        print(f"File {fdir_out}/{fname_out}.pkl exists. Skipping generation. Reading file...")
        df = pd.read_pickle(os.path.join(fdir_out, f"{fname_out}.pkl"))
        return df
    prompt_source_template = "{} was created by"
    prompt_target_template = "Who is the current CEO of {}"
    sample_id = 0

    print("Step 1: Prepare dataset...")
    records = []

    for key, value in multihop_samples.items():
        hop3, hop2 = key
        for hop1 in value:
            # hop1: Product
            # hop2: Company
            # hop3: CEO
            records.append({
                "sample_id": sample_id,
                "prompt_source": prompt_source_template.replace("{}", hop1),
                "position_source": -1,  # always doing next token prediction
                "prompt_target": prompt_target_template,
                "position_target": -1,

                "baseline_hop2": f"{hop1} was created by",  #  hop2
                "baseline_hop3": f"Who is the current CEO of {hop2}",  # hop3
                "baseline_multihop3": f"Who is the current CEO of the company that created {hop1}",  # hop3

                "hop1": hop1,
                "hop2": hop2,
                "hop3": hop3,
            })
            sample_id += 1

    # Step 2: Compute baseline generations
    print("Step 2: Compute baseline generations...")
    df = pd.DataFrame.from_records(records)
    eval_results = generate_baseline_multihop(mt, df, batch_size=batch_size, max_gen_len=max_gen_len)
    for key, value in eval_results.items():
        df[key] = list(value)

    df.to_csv(os.path.join(fdir_out, f"{fname_out}.tsv"), sep="\t")
    df.to_pickle(os.path.join(fdir_out, f"{fname_out}.pkl"))
    return df

def generate_comparison_multihop_data_generic(comparison_question_template, sub_question_template, comparison_multihop_question_template, name, df_samples: pd.DataFrame, fdir_out="./outputs/factual", batch_size=512 // batch_size_scale, max_gen_len=20, replace=True):
    print("batch size", batch_size)
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    fname_out = name
    if not replace and os.path.exists(f"{fdir_out}/{fname_out}.pkl"):
        print(f"File {fdir_out}/{fname_out}.pkl exists. Skipping generation. Reading file...")
        df = pd.read_pickle(os.path.join(fdir_out, f"{fname_out}.pkl"))
        return df
    q = "The CEO name which comes first alphabetically between the CEO of {} and the CEO of {} is"
    cq = "The CEO name which comes first alphabetically between {} and {}"
    # prompt_source_template = "{} was created by"
    # prompt_target_template = "Who is the current CEO of {}"
    sample_id = 0

    print("Step 1: Prepare dataset...")
    records = []
    for _, row in df_samples.iterrows():
    # for key, value in multihop_samples.items():
        subject1, object1, subject2, object2, answer = row 
        # hop3, hop2 = key
        # for hop1 in value:
        #     # hop1: Product
        #     # hop2: Company
        #     # hop3: CEO
        question_parts = comparison_multihop_question_template.split('{}')
        # question_parts[0] += subject1
        # question_parts[1] += subject2
        
        # inp1 = make_inputs(mt.tokenizer, [question_parts[0]], 'cpu')
        num_tokens1 = count_tokens(question_parts[0])
        
        # inp2 = make_inputs(mt.tokenizer, [question_parts[1]], 'cpu')
        # num_tokens2 = len(inp2["input_ids"]) 
        # num_tokens2 = count_tokens(question_parts[1])
        
        # inp3 = make_inputs(mt.tokenizer, [subject1], 'cpu')
        # num_token_subj1 = len(inp3["input_ids"]) 
        num_token_subj1 = count_tokens(subject1)
        num_token_subj2 = count_tokens(subject2)
        
        # inp4 = make_inputs(mt.tokenizer, ["between the country of"], 'cpu')
        # num_token_tmp = len(inp4["input_ids"])
        num_token_tmp  = count_tokens("the country of")
        num_tokens2 = count_tokens(comparison_multihop_question_template.split('{}?')[0].format(subject1)) 
        
        baseline_hop_list = [sub_question_template.format(subject1),
                                  sub_question_template.format(subject2),
                                  comparison_question_template.format(object1, object2)]
        hop_list = [object1, object2, answer]
        d = {
            "sample_id": sample_id,
            "prompt_source": comparison_multihop_question_template.format(subject1, subject2).replace('\n', '\\n'),
            # "position_sources": [num_tokens1 + num_token_subj1, -1],  # always doing next token prediction
            "position_sources": [num_tokens1 + num_token_subj1 - 3, num_tokens2 + num_token_subj2 - 3],
            "prompt_target": comparison_multihop_question_template.format(subject1, subject2).replace('\n', '\\n'),
            # "position_targets": [num_tokens1, num_tokens1 + num_tokens2 + num_token_subj1],
            "position_targets": [num_tokens1 - num_token_tmp, num_tokens2 - num_token_tmp],
            # "baseline_hop2": f"{hop1} was created by",  #  hop2
            # "baseline_hop3": f"Who is the current CEO of {hop2}",  # hop3
            # "baseline_multihop3": f"Who is the current CEO of the company that created {hop1}",  # hop3
            
            # "hop1": hop1,
            # "hop2": hop2,
            # "hop3": hop3,
            "baseline_multihop": comparison_multihop_question_template.format(subject1, subject2),
            
        }
        for i in range(len(baseline_hop_list)):
            d[f"baseline_hop_{i}"] = baseline_hop_list[i]
            d[f"hop_{i}"] = hop_list[i]
        records.append(d)
        sample_id += 1
    cases = [("baseline_hop_0", "hop_0"),
        ("baseline_hop_1", "hop_1"),
        ("baseline_hop_2", "hop_2"),
        ("baseline_multihop", "hop_2")
        ]
    # Step 2: Compute baseline generations
    print("Step 2: Compute baseline generations...")
    df = pd.DataFrame.from_records(records)
    eval_results = generate_baseline_multihop(mt, df, batch_size=batch_size, max_gen_len=max_gen_len, cases_list=cases)
    for key, value in eval_results.items():
        df[key] = list(value)
    for col, _ in cases:
        df[col] = df[col].apply(lambda x: x.replace('\n', '\\n'))
        
    df.to_csv(os.path.join(fdir_out, f"{fname_out}.tsv"), sep="\t")
    df.to_pickle(os.path.join(fdir_out, f"{fname_out}.pkl"))
    
    correct_subset = df[df["is_correct_baseline_hop_0"]].reset_index(drop=True)
    correct_subset = correct_subset[correct_subset["is_correct_baseline_hop_1"]].reset_index(drop=True)
    correct_subset = correct_subset[correct_subset["is_correct_baseline_hop_2"]].reset_index(drop=True)
    correct_subset.to_csv(os.path.join(fdir_out, f"{fname_out}_only_correct_True.tsv"), sep="\t")
    correct_subset.to_pickle(os.path.join(fdir_out, f"{fname_out}_only_correct_True.pkl"))
    return df

# %%
# multihop_df = generate_multihop_data_ceo(batch_size=128 // batch_size_scale, max_gen_len=20)
# multihop_comp_q = "The CEO name which comes first alphabetically between the CEO of {} and the CEO of {} is"
multihop_comp_q = "The country name which comes first alphabetically between Spain and Russia?\nAnswer: Russia\n\nThe country name which comes first alphabetically between India and Malaysia?\nAnswer: India\n\nThe country name which comes first alphabetically between the country of {} and the country of {}?\nAnswer:"
# multihop_comp_q = "The country name which comes first alphabetically between Spain and Russia?\nAnswer: Russia\n\nThe country name which comes first alphabetically between the country of {} and the country of {}?\nAnswer:"

# multihop_comp_q = "What is the name which comes first alphabetically between the name of the CEO of {} and the name of the CEO of {}"
# comp_q = "The CEO name which comes first alphabetically between {} and {}"
# comp_q = "The CEO name which comes first alphabetically between Sundar Pichai and Elon Musk?\n Answer: Elon Musk\n\n The CEO name which comes first alphabetically between Jensen Huang and Tim Cook?\n Answer: Jensen Huang\n\nThe CEO name which comes first alphabetically between {} and {}? Answer:"
comp_q = "The country name which comes first alphabetically between Spain and Russia?\nAnswer: Russia\n\nThe country name which comes first alphabetically between India and Malaysia?\nAnswer: India\n\nThe country name which comes first alphabetically between {} and {}?\nAnswer:"

# comp_q = "The country name which comes first alphabetically between Spain and Russia?\nAnswer: Russia\n\nThe country name which comes first alphabetically between {} and {}?\nAnswer:"
sub_question = "What country is {} in? It is in "

# "The CEO name which comes first alphabetically between {} and {}""
# comp_q = "What is the name which comes first alphabetically between {} and {}"
# sample_df = pd.read_csv('landmark_country_comparison.csv').sample(64 * 200)
# sub_question = "Who is the current CEO of {}"
# !!!!!!!!!!!!!!!!!!!!
# generate_comparison_multihop_data_generic(comp_q, sub_question, multihop_comp_q, "multihop_landmark_country_comparison_two_shot_two_patches_small", sample_df)

# %%
def evaluate_attriburte_exraction_batch_multihop(
        mt, df, batch_size=256 // batch_size_scale, max_gen_len=10, transform=None, patch_count=1, two_generations=False
):
    def _evaluate_attriburte_exraction_single_batch(batch_df):
        batch_size = len(batch_df)
        prompt_source_batch = np.array([x.replace('\\n', '\n') for x in batch_df["prompt_source"]])
        prompt_target_batch = np.array([x.replace('\\n', '\n') for x in batch_df["prompt_target"]])
        # prompt_target_batch = np.array(batch_df["prompt_target"]).apply(lambda x: replace('\\n', '\n'))
        
        if "layer_sources" in batch_df:
            layer_sources_batch = np.array(batch_df["layer_sources"])
            layer_targets_batch = np.array(batch_df["layer_targets"])
        else:
            # backwards compatibility
            layer_sources_batch = np.expand_dims(np.array(batch_df["layer_source"]), -1)
            layer_targets_batch = np.expand_dims(np.array(batch_df["layer_target"]), -1)
            
        if "position_sources" in batch_df:
            # print("in position sources")
            # print(batch_df["position_sources"])
            position_sources_batch = np.array(batch_df["position_sources"])
            position_targets_batch = np.array(batch_df["position_targets"])
        else:
            position_sources_batch = np.expand_dims(np.array(batch_df["position_source"]), -1)
            position_targets_batch = np.expand_dims(np.array(batch_df["position_target"]), -1)

        object_batch = np.array(batch_df["hop_2"])

        # Adjust position_target to be absolute rather than relative
        inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
        for i in range(batch_size):
            for j in range(patch_count):
                if position_targets_batch[i][j] < 0:
                    position_targets_batch[i][j] += len(inp_target["input_ids"][j])

        # Step 1: run the the model on source without patching and get the hidden representations.
        inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
        output_orig = mt.model(**inp_source, output_hidden_states=True)

        # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
        hidden_reps = [
            [
                output_orig.hidden_states[layer_sources_batch[i][j] + 1][i][position_sources_batch[i][j]]
                for i in range(batch_size)
            ]
            for j in range(patch_count)
        ]
        if transform is not None:
            for i in range(patch_count):
                for j in range(batch_size):
                    hidden_reps[i][j] = transform(hidden_reps[i][j])
        
        # Step 2: do second run on target prompt, while patching the input hidden state.
        hs_patch_configs = [
            [
                {
                    "batch_idx": i,
                    "layer_target": layer_targets_batch[i][j],
                    "position_target": position_targets_batch[i][j],
                    "hidden_rep": hidden_reps[j][i],  # supposed to be reversed indices
                    "skip_final_ln": (
                            layer_sources_batch[i][j]
                            == layer_targets_batch[i][j]
                            == mt.num_layers - 1
                    ),
                }
                for i in range(batch_size)
            ]
            for j in range(patch_count) 
        ]
        gen_mode = not(two_generations)
        patch_hooks_list = [
            mt.set_hs_patch_hooks(mt.model, hs_patch_config, patch_input=False, generation_mode=gen_mode)
            for hs_patch_config in hs_patch_configs
        ]
        
        
        if two_generations:
            output_first_gen = mt.model(**inp_source)
            for patch_hooks in patch_hooks_list:
                remove_hooks(patch_hooks)
            hidden_reps = [
            [
                output_first_gen.hidden_states[layer_sources_batch[i][j] + 1][i][position_sources_batch[i][j]]
                for i in range(batch_size)
            ]
            for j in range(patch_count)
                    ]
            if transform is not None:
                for i in range(patch_count):
                    for j in range(batch_size):
                        hidden_reps[i][j] = transform(hidden_reps[i][j])
            
            # Step 2: do second run on target prompt, while patching the input hidden state.
            hs_patch_configs = [
                [
                    {
                        "batch_idx": i,
                        "layer_target": layer_targets_batch[i][j],
                        "position_target": position_targets_batch[i][j],
                        "hidden_rep": hidden_reps[j][i],  # supposed to be reversed indices
                        "skip_final_ln": (
                                layer_sources_batch[i][j]
                                == layer_targets_batch[i][j]
                                == mt.num_layers - 1
                        ),
                    }
                    for i in range(batch_size)
                ]
                for j in range(patch_count) 
            ]
            patch_hooks_list = [
            mt.set_hs_patch_hooks(mt.model, hs_patch_config, patch_input=False, generation_mode=gen_mode)
            for hs_patch_config in hs_patch_configs
            ]
            
            

        # NOTE: inputs are left padded,
        # and sequence length is the same across batch
        seq_len = len(inp_target["input_ids"][0])
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=seq_len + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[:, seq_len:]
        generations_patched = decode_tokens(mt.tokenizer, output_toks)
        generations_patched_txt = np.array([
            " ".join(generations_patched[i])
            for i in range(batch_size)
        ])
        is_correct_patched = np.array([
            (object_batch[i] in generations_patched_txt[i]
             or object_batch[i].replace(" ", "") in generations_patched_txt[i].replace(" ", ""))
            for i in range(batch_size)
        ])

        # remove patching hooks
        for patch_hooks in patch_hooks_list:
            remove_hooks(patch_hooks)


        cpu_hidden_reps = [np.array([hidden_rep[i].detach().cpu().numpy() for i in range(batch_size)]) for hidden_rep in hidden_reps]

        # results_list = [{
        #     "generations_patched": generations_patched,
        #     "is_correct_patched": is_correct_patched,
        #     "hidden_rep": cpu_hidden_rep,
        # } for cpu_hidden_rep in cpu_hidden_reps]
        results = {
            "generations_patched": generations_patched,
            "is_correct_patched": is_correct_patched,
        }

        return results

    # patch_results = [{}] * patch_count
    # n_batches = len(df) // batch_size
    # if len(df) % batch_size != 0:
    #     n_batches += 1
    # for i in tqdm.tqdm(range(len(df) // batch_size), desc='patching experiment iteration'):
    #     cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
    #     batch_results = _evaluate_attriburte_exraction_single_batch(cur_df)
    #     for patch_idx in range(patch_count): 
    #         for key, value in batch_results[patch_idx].items():
    #             if key in patch_results[patch_idx]:
    #                 patch_results[patch_idx][key] = np.concatenate((patch_results[patch_idx][key], value))
    #             else:
    #                 patch_results[patch_idx][key] = value
    
    patch_results = {} 
    n_batches = len(df) // batch_size
    if len(df) % batch_size != 0:
        n_batches += 1
    for i in tqdm.tqdm(range(n_batches), desc='patching experiment iteration'):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_results = _evaluate_attriburte_exraction_single_batch(cur_df)
        for key, value in batch_results.items():
            if key in patch_results:
                patch_results[key] = np.concatenate((patch_results[key], value))
            else:
                patch_results[key] = value

    return patch_results

# %%
def run_experiment(fname_in, fdir_out, fname_out="multihop", batch_size=512 // batch_size_scale, n_samples=-1,
                   save_output=True, replace=False, tsv=False, patch_count=1, is_identical_layers=True, is_src_gt_dst=True, only_optimal=False):
    # patch_count is the number of compared entities. in the original experiment, it's 1 and in our case it's 2.
    print(f"Running experiment on {fname_in}...")
    print("output_path:",f"{fdir_out}/{fname_out}.pkl")
    print(f"Patch count {patch_count}")
    if not replace and os.path.exists(f"{fdir_out}/{fname_out}.pkl"):
        print(f"File {fdir_out}/{fname_out}.pkl exists. Skipping generation. Reading file...")
        results_df = pd.read_pickle(f"{fdir_out}/{fname_out}.pkl")
        return results_df
    if tsv:
        df = pd.read_csv(f"{fname_in}", sep='\t', header=0)
        if 'position_sources' in df.columns:
            print("converted_position sources")
            df['position_sources'] = df['position_sources'].apply(json.loads)
            df['position_targets'] = df['position_targets'].apply(json.loads) 
    else:
        df = pd.read_pickle(f"{fname_in}")
    
        
    # BATCHing all layers combinations
    batch = []
    layer_sources= np.arange(mt.num_layers)
    layer_targets = np.arange(mt.num_layers)    
    if is_identical_layers:
        layers_meshed = np.transpose(np.meshgrid(layer_sources, layer_targets)).reshape(-1, 2)
        layers_ds = np.expand_dims(layers_meshed, -1).repeat(patch_count, axis=-1)
    else: 
        mesh_list = [layer_sources, layer_targets] * patch_count
        layers_ds = np.reshape(np.meshgrid(*mesh_list), (patch_count, 2, -1)).transpose()
    if is_src_gt_dst:
        # reduces amount by (2*38/37)^2 ~= 4.2
        layers_ds = layers_ds[[np.all(layer[0] > layer[1]) for layer in layers_ds]]
    #  TODO: support further limitations on src/dst
    if only_optimal:
        layers_ds = np.expand_dims(np.array([14, 0]).reshape(1,2), -1).repeat(patch_count, axis=-1)
        # layers_meshed = np.transpose(np.meshgrid(np.arange(10, 15), np.arange(3))).reshape(-1, 2)
        # layers_ds = np.expand_dims(layers_meshed, -1).repeat(patch_count, axis=-1)
    if n_samples > 0:
        df = df.sample(n=n_samples // (layers_ds.shape[0]), replace=False, random_state=42).reset_index(drop=True)
    
    print(f"\tNumber of samples: {len(df)}")
    
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f'Building experiments dataframe'):
        for layers in layers_ds:
                item = dict(row)
                item.update({
                    "layer_sources": layers[0],
                    "layer_targets": layers[1],
                })
                if layers[0].shape[0] == 1:
                    item.update({"layer_source": layers[0]})
                
                if layers[1].shape[0] == 1:
                    item.update({"layer_target": layers[1]})
                    
                batch.append(item)
    experiment_df = pd.DataFrame.from_records(batch)

    # if n_samples > 0 and n_samples < len(experiment_df):
    #     experiment_df = experiment_df.sample(n=n_samples, replace=False, random_state=42).reset_index(drop=True)

    print(f"\tNumber of datapoints for patching experiment: {len(experiment_df)}")

    eval_results = evaluate_attriburte_exraction_batch_multihop(mt, experiment_df, batch_size=batch_size, patch_count=patch_count)

    # eval_results = eval_results  # TODO: add support for the multi-patch 
    print("head debug")
    print(len(eval_results["is_correct_patched"]))
    print("len experiments df", len(experiment_df))
    results_df = experiment_df.head(len(eval_results["is_correct_patched"]))
    print("len results df", len(results_df))

    for key, value in eval_results.items():
        results_df[key] = list(value)
    results_df['generations_patched'] = results_df['generations_patched'].apply(lambda x: np.array2string(x, separator=', '))
    results_df['generations_patched'] = results_df['generations_patched'].apply(lambda x: x.replace('\n', '\\n'))
    if save_output:
        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)
        results_df.to_csv(f"{fdir_out}/{fname_out}.tsv", sep="\t")
        results_df.to_pickle(f"{fdir_out}/{fname_out}.pkl")

    return results_df

# # %%
# run_experiment("./outputs/factual/multihop_product_company_ceo.pkl",
#                "./outputs/results/factual",
#                fname_out="multihop_product_company_ceo", batch_size=128 // batch_size_scale, n_samples=-1,
#                save_output=True, replace=True)

# %% [markdown]
# # Probe

# %%
def probe_baseline(task_type="factual", task_name="multihop_product_company_ceo",
                   fname_input="./outputs/factual/multihop_product_company_ceo.pkl",
                   inp_label_name="object",
                   hidden_states_dir="./outputs/results_ceo",
                   probe_res_dir="./outputs/probe_ceo",
                   label_name="hop3", seed=42, n_test_samples=2,  # test_ratio=0.5, n_samples=4,
                   rewrite=False, only_correct=True):
    fdir = f"{probe_res_dir}/{task_type}"
    fname_pkl = f"{fdir}/{task_name}_only_correct_{only_correct}.pkl"
    if rewrite == False and os.path.exists(fname_pkl):
        print(f"\t{fname_pkl} exists. Skipping generation. Reading file...")
        test_df = pd.read_pickle(fname_pkl)
        return test_df
    print(f"Creating {fname_pkl}...")
    np.random.seed(seed)

    fname_hidden_states = f"{hidden_states_dir}/{task_type}/{task_name}.pkl"

    # Retrieve list of classes from inputs
    inps_df = pd.read_csv(fname_input, sep='\t', header=0)
    classes = np.unique(inps_df[inp_label_name])
    classes_dict = {}
    for idx, cls in enumerate(classes):
        classes_dict[cls] = idx

    # Get saved hiddens
    hiddens_df = pd.read_pickle(fname_hidden_states)
    hiddens_df = hiddens_df.sample(frac=1).reset_index(drop=True)
    if only_correct:
        hiddens_df = hiddens_df[hiddens_df["is_correct_baseline_hop2"]].reset_index(drop=True)
        hiddens_df = hiddens_df[hiddens_df["is_correct_baseline_hop3"]].reset_index(drop=True)
        if len(hiddens_df) < 1:
            print(f'\tNo correct predictions for {fname_pkl}. Skipping...')
            return
    sample_ids = np.unique(hiddens_df['sample_id'])
    if len(sample_ids) < 4:
        print(f"\tNot enough samples to train a probe for {fname_pkl}. Skipping...")
        return
    np.random.shuffle(sample_ids)
    test_sample_ids = sample_ids[:n_test_samples]
    train_sample_ids = sample_ids[n_test_samples:]
    train_df = hiddens_df[hiddens_df['sample_id'].isin(train_sample_ids)]
    test_df = hiddens_df[hiddens_df['sample_id'].isin(test_sample_ids)]
    xs = np.stack(hiddens_df["hidden_rep"])
    ys = np.array([classes_dict[i] for i in hiddens_df[label_name]])

    train_xs = np.stack(train_df["hidden_rep"])
    train_ys = np.array([classes_dict[i] for i in train_df[label_name]])
    if len(np.unique(train_ys)) < 2:
        print(f"\tNot enough variety to train a probe for {fname_pkl}. Skipping...")
        return
    test_xs = np.stack(test_df["hidden_rep"])
    test_ys = np.array([classes_dict[i] for i in test_df[label_name]])

    clf = LogisticRegression(random_state=seed).fit(train_xs, train_ys)
    predicted_ys = clf.predict(test_xs)
    test_df["object_int"] = test_ys
    test_df["predicted_int"] = predicted_ys
    test_df["predicted"] = classes[predicted_ys]
    test_df["is_correct_probe"] = test_ys == predicted_ys

    if not os.path.exists(fdir):
        os.makedirs(fdir)
    test_df.to_csv(os.path.join(fdir, f"{task_name}_only_correct_{only_correct}.tsv"), sep="\t")
    test_df.to_pickle(os.path.join(fdir, f"{task_name}_only_correct_{only_correct}.pkl"))
    return test_df

# %%
# probe_baseline(task_type="factual", task_name="multihop_product_company_ceo",
#                # fname_input="./preprocessed_data/factual/company_ceo.pkl",
#                fname_input="./preprocessed_data/factual/company_ceo.tsv",
#                inp_label_name="object",
#                hidden_states_dir="./outputs/results",
#                probe_res_dir="./outputs/probe_ceo",
#                label_name="hop3",
#                rewrite=False)

# %% [markdown]
# # Plots

# %%
def plot_heatmaps(task_type="factual", task_name="multihop_product_company_ceo", version="v5", _vmin=0, _vmax=1):
    probe_res_fname = f"./outputs/probe_{version}/{task_type}/{task_name}_only_correct_True.pkl"
    probe_df = pd.read_pickle(probe_res_fname)
    plot_ttl = f"{task_type} : {task_name} - {model_name.strip('./')}"
    n_samples = len(probe_df)

    heatmap_patch = probe_df.groupby(['layer_target', 'layer_source'])["is_correct_patched"].mean().unstack()
    ax = sns.heatmap(data=heatmap_patch, cmap="crest_r", vmin=_vmin, vmax=_vmax)
    ax.invert_yaxis()
    ax.set_title(f"{plot_ttl} \npatch accuracy (# samples: {n_samples})")
    plt.show()
    plt.clf()

    ax = sns.lineplot(data=probe_df, x="layer_source", y="is_correct_probe")
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(f"{plot_ttl} \nprobe accuracy (# samples: {n_samples})")
    plt.show()
    plt.clf()

# %%
def plot_patching_heatmaps(task_type="factual", task_name="multihop_product_company_ceo", version="ceo",
                           _vmin=0, _vmax=1):
    patch_res_fname = f"./outputs/results/{task_type}/{task_name}.pkl"
    patch_df = pd.read_pickle(patch_res_fname)
    patch_df = patch_df[patch_df["is_correct_baseline_hop2"]].reset_index(drop=True)
    patch_df = patch_df[patch_df["is_correct_baseline_hop3"]].reset_index(drop=True)
    n_samples = len(patch_df)
    if n_samples == 0:
        print(f"No correct predictions for {patch_res_fname}. Skipping...")
        return
    plot_ttl = f"{task_type}: {task_name}\n{model_name.strip('./')}"
    baseline_acc_multihop3 = patch_df["is_correct_baseline_multihop3"].mean() * 100
    baseline_acc_hop3 = patch_df["is_correct_baseline_hop3"].mean() * 100
    baseline_acc_hop2 = patch_df["is_correct_baseline_hop2"].mean() * 100

    heatmap_patched = patch_df.groupby(['layer_target', 'layer_source'])["is_correct_patched"].mean().unstack()
    ax = sns.heatmap(data=heatmap_patched, cmap="crest_r", vmin=_vmin, vmax=_vmax)
    ax.invert_yaxis()
    ax.set_title(
        f"{plot_ttl}\nPatching accuracy\nBaseline multihop reasoning accuracy: {baseline_acc_multihop3:.2f}\n(# samples: {n_samples})")
    plt.show()
    plt.clf()

# %%
# plot_patching_heatmaps(version="ceo")

# %%
# plot_heatmaps(version="ceo")

# %% [markdown]
# # Experiment 2 : CoT experiment subset
# 
# This is a subset made only from (product, company) and (company, CEO) tuples from the LRE dataset.
# We only picked 3 (company, CEO) tuples, and 15 (product, company) tuples for each that the model is more likely to know the answer to.
# 
# This is an exploratory experiment. There is a more complete experiment later in the colab.
# Hop 1: Product
# Hop 2: company
# Hop 3: CEO
# 
# The difference between this and experiment 1 is in the choice of source and target prompt template. In this experiment, concatenation of source and target prompt makes a reasonable query, compared to experiment 1 where they where that wasn't the case.

# %%
multihop_samples = {
    ("Satya Nadella", "Microsoft"): ["WinDbg", ".NET Framework", "Internet Explorer", "MS-DOS", "Office Open XML",
                                     "TypeScript", "Bing Maps Platform", "Outlook Express", "PowerShell", "Windows 95",
                                     "Xbox 360", "Zune", "Visual Basic Script", "Virtual Hard Disk", "Robocopy",
                                     ],
    ("Tim Cook", "Apple"): ["Siri", "App Store", "CarPlay", "MacBook Air", "Xcode",
                            "macOS", "iWork", "Safari", "QuickTime", "TextEdit",
                            "WebKit", "QuickDraw", "Time Machine (macOS)", "MessagePad", "Macbook Pro",
                            ],
    ("Sundar Pichai", "Google"): ["Chromecast", "Chromebook", "Wear OS", "G Suite", "Picasa",
                                  "WebP Lossless", "General Transit Feed Specification Lossless", "Cloud Spanner",
                                  "Android TV", "Android Runtime",
                                  "Android Jelly Bean", "Android Auto", "App Inventor", "Chromebook Pixel",
                                  "Project Ara",
                                  ]
}


def generate_CoT_data_prod(fdir_out="./outputs/preprocessed_data_prod_CoT/factual", batch_size=512 // batch_size_scale,
                           max_gen_len=20):
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    prompt_source_template = "Who is the current CEO of "
    prompt_target_template = "the company that created {}"
    sample_id = 0

    print("Step 1: Prepare dataset...")
    records = []

    for key, value in multihop_samples.items():
        hop3, hop2 = key
        for hop1 in value:
            # hop1: Product
            # hop2: Company
            # hop3: CEO

            records.append({
                "sample_id": sample_id,
                "prompt_source": prompt_source_template,
                "position_source": -1,  # always doing next token prediction
                "prompt_target": prompt_target_template.replace("{}", hop1),
                "position_target": -1,

                "baseline_hop2": f"the company that created {hop1}",  #  hop2
                "baseline_hop3": f"Who is the current CEO of {hop2}",  # hop3
                "baseline_multihop3": f"Who is the current CEO of the company that created {hop1}",  # hop3

                "hop1": hop1,
                "hop2": hop2,
                "hop3": hop3,
            })
            sample_id += 1

    # Step 2: Compute baseline generations
    print("Step 2: Compute baseline generations...")
    df = pd.DataFrame.from_records(records)
    eval_results = generate_baseline_multihop(mt, df, batch_size=batch_size, max_gen_len=max_gen_len)
    for key, value in eval_results.items():
        df[key] = list(value)

    df.to_csv(os.path.join(fdir_out, "multihop_product_company_ceo.tsv"), sep="\t")
    df.to_pickle(os.path.join(fdir_out, "multihop_product_company_ceo.pkl"))

    correct_subset = df[df["is_correct_baseline_hop2"]].reset_index(drop=True)
    correct_subset = correct_subset[correct_subset["is_correct_baseline_hop3"]].reset_index(drop=True)
    correct_subset.to_csv(os.path.join(fdir_out, "multihop_product_company_ceo_only_correct_True.tsv"), sep="\t")
    correct_subset.to_pickle(os.path.join(fdir_out, "multihop_product_company_ceo_only_correct_True.pkl"))
    return df

# %%
# multihop2_df = generate_CoT_data_prod(batch_size=128 // batch_size_scale, max_gen_len=20)

# %%
# cot_correct_baseline = run_experiment(
#     "./outputs/preprocessed_data_prod_CoT/factual/multihop_product_company_ceo_only_correct_True.pkl",
#     "./outputs/results_prod_CoT/factual",
#     fname_out = "multihop_product_company_ceo_only_correct_True", batch_size=128 // batch_size_scale, n_samples=-1,
#     save_output=True, replace=False)

# %%
# print("Base MultiHop Accuracy: ",
#       cot_correct_baseline.groupby(['sample_id'])["is_correct_baseline_multihop3"].max().reset_index()["is_correct_baseline_multihop3"].mean())
# 
# print("Patching MultiHop Accuracy: ",
#       cot_correct_baseline.groupby(['sample_id'])["is_correct_patched"].max().reset_index()["is_correct_patched"].mean())

# %% [markdown]
# # Experimet 3: Main CoT experiment
# 
# This is the full version, using maximal amount of data possible from LRE where a multihop question can be formed combining two single-hop questions.

# %%
def generate_CoT_data_v7(fname_in="./outputs/preprocessed_data_LRE_CoT/factual_multihop/combined_multihop.pkl",
                         fdir_out="./outputs/preprocessed_data_LRE_CoT/factual_multihop",
                         batch_size=512 // batch_size_scale, max_gen_len=20):
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    fname_pkl = f"{fdir_out}/combined_multihop_CoT_{model_name}_only_correct_True.pkl"
    if os.path.exists(fname_pkl):
        print(f"File {fname_pkl} exists. Skipping...")
        return

    print("Step 1: Read multihop dataset created using LRE data prep...")
    df = pd.read_pickle(fname_in)

    # Step 2: Compute baseline generations
    print("Step 2: Compute baseline generations...")
    eval_results = generate_baseline_multihop(mt, df, batch_size=batch_size, max_gen_len=max_gen_len)
    for key, value in eval_results.items():
        df[key] = list(value)

    df.to_csv(os.path.join(fdir_out, f"combined_multihop_CoT_{model_name}.tsv"), sep="\t")
    df.to_pickle(os.path.join(fdir_out, f"combined_multihop_CoT_{model_name}.pkl"))

    correct_subset = df[df["is_correct_baseline_hop2"]].reset_index(drop=True)
    correct_subset = correct_subset[correct_subset["is_correct_baseline_hop3"]].reset_index(drop=True)
    correct_subset.to_csv(os.path.join(fdir_out, f"combined_multihop_CoT_{model_name}_only_correct_True.tsv"), sep="\t")
    correct_subset.to_pickle(os.path.join(fdir_out, f"combined_multihop_CoT_{model_name}_only_correct_True.pkl"))
    return df

# %%
# generate_CoT_data_v7(fname_in="./outputs/preprocessed_data_LRE_CoT/factual_multihop/combined_multihop.pkl",
#                      fdir_out="./outputs/preprocessed_data_LRE_CoT/factual_multihop",
#                      batch_size=128 // batch_size_scale, max_gen_len=20)



# %%
# cot_correct_baseline = run_experiment(
#     f"./preprocessed_data/factual_multihop/multihop_CoT_vicuna-13b-v1.1.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_CoT_{model_name}_debug", batch_size=128, n_samples=40 ** 2 * 500,
#     save_output=False, replace=True, tsv=True, patch_count=1)
# !!!!!!!!!!!!!!!!

# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison2_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True", batch_size=128 // 4, n_samples=-1,
#     save_output=True, replace=False, tsv=True, is_src_gt_dst=False)
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_one_hop_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True", batch_size=128 // 4, n_samples=-1,
#     save_output=True, replace=False, tsv=True)

# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_hop", batch_size=128 // 4, n_samples=32 * 12000,
#     save_output=True, replace=False, tsv=True, is_src_gt_dst=False)
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_two_patches", batch_size=128 // 4, n_samples=32 * 10,
#     save_output=True, replace=False, tsv=True, is_src_gt_dst=False, patch_count=2)
## one patch experiment
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_small_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_one_patch", batch_size=128 // 4, n_samples=32 * 24000,
#     save_output=True, replace=False, tsv=True, is_src_gt_dst=False, patch_count=1)
# # Two patches experiment
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_small_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_two_patches", batch_size=128 // 4, n_samples=32 * 24000,
#     save_output=True, replace=False, tsv=True, is_src_gt_dst=False, patch_count=2)
## one patch experiment only 0 14:
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_small_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_one_patch_only_one_combination", batch_size=128 // 4, n_samples=32 * 15 * 5 * 3,
#     save_output=True, replace=True, tsv=True, is_src_gt_dst=False, patch_count=1, only_optimal=True)
# two patch experiment only 0 14:

# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_small_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_two_patch_only_one_combination", batch_size=128 // 4, n_samples=-1,
#     save_output=True, replace=True, tsv=True, is_src_gt_dst=False, patch_count=2, only_optimal=True)
# cot_correct_baseline = run_experiment(
#     f"./outputs/factual/multihop_landmark_country_comparison_two_shot_two_patches_small_only_correct_True.tsv",
#     "./outputs/preprocessed_data/factual_multihop",
#     fname_out=f"combined_multihop_comparison_CoT_{model_name}_only_correct_True_two_shot_two_patches_test", batch_size=128 // 4, n_samples=32 * 10000,
#     save_output=True, replace=True, tsv=True, is_src_gt_dst=False, patch_count=2, only_optimal=False)


# 23000
print("Finished Experiment now trying to plot heatmaps")
# %%
# efficient_subset = cot_correct_baseline[
#     cot_correct_baseline["layer_source"] > cot_correct_baseline["layer_target"]].reset_index(drop=True)
# TODO maybe run patching for all source x target, but the killer case is when source < target

print("Base MultiHop Accuracy: ",
      cot_correct_baseline.groupby(['sample_id'])["is_correct_baseline_multihop"].max().reset_index()["is_correct_baseline_multihop"].mean())

print("General Patching MultiHop Accuracy (all source layer x target layer): ",
      cot_correct_baseline.groupby(['sample_id'])["is_correct_patched"].max().reset_index()[
          "is_correct_patched"].mean())

# print("Efficient Patching MultiHop Accuracy (source layer < target layer): ",
#       efficient_subset.groupby(['sample_id'])["is_correct_patched"].max().reset_index()["is_correct_patched"].mean())


# %%
# multihop_fname = "./preprocessed_data/factual_multihop/multihop_CoT_vicuna-13b-v1.1.tsv"
# df = pd.read_csv(multihop_fname, sep='\t', header=0)
# print(len(df))

# multihop_fname_only_correct = f"./outputs/preprocessed_data/factual_multihop/combined_multihop_CoT_{model_name}_only_correct_True.pkl"
# df_only_correct = pd.read_pickle(multihop_fname_only_correct)
# print(len(df_only_correct))
# df_only_correct.groupby(['fname_src', 'fname_target']).count()

# %%
def plot_patching_heatmaps_from_df(patch_df, _vmin=0, _vmax=None, fname_postfix="", save_output=True):
    n_samples = len(patch_df)
    plots_dir = "./outputs/multihop_reasoning"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # baseline_acc_multihop3 = patch_df["is_correct_baseline_multihop3"].mean()*100
    # patching_acc = patch_df.groupby(['sample_id'])["is_correct_patched"].max().reset_index()["is_correct_patched"].mean() * 100
    patch_df['layer_source'] = patch_df['layer_sources'].apply(lambda x: x[0])
    patch_df['layer_target'] = patch_df['layer_targets'].apply(lambda x: x[0])
    patch_df = patch_df[patch_df['is_correct_baseline_multihop'] == False]
    heatmap_patched = patch_df.groupby(['layer_target', 'layer_source'])["is_correct_patched"].mean().unstack()

    FONT_SIZE_TITLE = 16
    FONT_SIZE_AXIS = 15

    plt.figure()
    ax = sns.heatmap(data=heatmap_patched, cmap="crest_r", vmin=_vmin, vmax=_vmax)
    ax.invert_yaxis()
    ax.set_title(f"Self-correction in Multi-hop Reasoning\n# samples: {n_samples}", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Source Layer ($\ell$)", fontsize=FONT_SIZE_AXIS)
    plt.ylabel("Target Layer ($\ell^*$)", fontsize=FONT_SIZE_AXIS)
    plt.tight_layout()
    if save_output:
        fname = f"{plots_dir}/multihop_heatmap{fname_postfix}.pdf"
        plt.savefig(fname, format="pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{fname[:-4]}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

# %%
# plot_patching_heatmaps_from_df(efficient_subset, fname_postfix="_source_smaller_than_target")

# %%
plot_patching_heatmaps_from_df(cot_correct_baseline)

# %% [markdown]
# # Experiment 4 - CoT Let's think step by step baseline. Baseline
# 
# How does a "Let's think step by step" CoT baseline compare with the CoT Patchscope?

# %%
def step_by_step_cot_baseline(
        fname_in=f"./preprocessed_data/factual_multihop/combined_multihop_CoT_{model_name}_only_correct_True.pkl",
        fdir_out="./outputs/results_CoT/factual_multihop",
        fname_out=f"combined_multihop_CoT_{model_name}_only_correct_True_step_by_step",
        batch_size=128 // batch_size_scale,
        max_gen_len=20,
        rewrite=False,
        target_col="baseline_multihop3",
        object_col="hop3",
        cot_prefix="Let's think step by step. "):
    if not os.path.exists(fname_in):
        print(f'File {fname_in} does not exist. Skipping...')
        return

    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    fname_pkl = f"{fdir_out}/{fname_out}.pkl"
    if rewrite == False and os.path.exists(fname_pkl):
        print(f"\t{fname_pkl} exists. Skipping generation. Reading file...")
        df = pd.read_pickle(fname_pkl)
        return df

    print("Computing step-by-step baseline generations...")
    df = pd.read_pickle(fname_in)
    df["cot_prefix"] = cot_prefix

    def _generate_baseline_single_batch(batch_df):
        batch_size = len(batch_df)

        results = {}
        target_baseline_batch = np.array(batch_df[target_col])
        target_baseline_batch = np.core.defchararray.add(cot_prefix, target_baseline_batch.astype(str))
        object_batch = np.array(batch_df[object_col])

        inp_target_baseline = make_inputs(mt.tokenizer, target_baseline_batch, mt.device)
        seq_len_target_baseline = len(inp_target_baseline["input_ids"][0])
        output_target_baseline_toks = mt.model.generate(
            inp_target_baseline["input_ids"],
            max_length=seq_len_target_baseline + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[:, seq_len_target_baseline:]
        generations_baseline = decode_tokens(mt.tokenizer, output_target_baseline_toks)
        generations_baseline_txt = np.array([" ".join(sample_gen) for sample_gen in generations_baseline])

        is_correct_baseline = np.array([
            (object_batch[i] in generations_baseline_txt[i] or
             object_batch[i].replace(" ", "") in generations_baseline_txt[i].replace(" ", ""))
            for i in range(batch_size)
        ])
        results.update(
            {
                f"step_by_step_generations_{target_col}": generations_baseline_txt,
                f"step_by_step_is_correct_{target_col}": is_correct_baseline,
            }
        )

        return results

    results = {}
    n_batches = len(df) // batch_size
    if len(df) % batch_size != 0:
        n_batches += 1
    for i in tqdm.tqdm(range(n_batches)):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_results = _generate_baseline_single_batch(cur_df)
        for key, value in batch_results.items():
            if key in results:
                results[key] = np.concatenate((results[key], value))
            else:
                results[key] = value

    for key, value in results.items():
        df[key] = list(value)

    df.to_csv(os.path.join(fdir_out, f"{fname_out}.tsv"), sep="\t")
    df.to_pickle(os.path.join(fdir_out, f"{fname_out}.pkl"))
    return df

# %%
# cot_correct_baseline_step_by_step_baseline = step_by_step_cot_baseline(
#     fname_in=f"./outputs/preprocessed_data/factual_multihop/combined_multihop_CoT_{model_name}_only_correct_True.pkl",
#     fdir_out="./outputs/results_LRE_CoT/factual_multihop",
#     fname_out=f"combined_multihop_CoT_{model_name}_only_correct_True_step_by_step",
#     batch_size=128 // batch_size_scale,
#     max_gen_len=20,
#     target_col="baseline_multihop3",
#     object_col="hop3",
#     cot_prefix="Let's think step by step. ",
#     rewrite=True)

# %%
# print("Base MultiHop Accuracy: ",
#       cot_correct_baseline.groupby(['sample_id'])["is_correct_baseline_multihop3"].max().reset_index()["is_correct_baseline_multihop3"].mean())

# print("General Patching MultiHop Accuracy (all source layer x target layer): ",
#       cot_correct_baseline.groupby(['sample_id'])["is_correct_patched"].max().reset_index()[
#           "is_correct_patched"].mean())

# print("Canonical CoT ('Let's think step by step. ') MultiHop Accuracy: ",
#       cot_correct_baseline_step_by_step_baseline.groupby(['sample_id'])[
#           "step_by_step_is_correct_baseline_multihop3"].max().reset_index()[
#           "step_by_step_is_correct_baseline_multihop3"].mean())

# %%
# cot_correct_baseline_step_by_step_baseline['step_by_step_generations_baseline_multihop3']
# cot_correct_baseline_step_by_step_baseline[
#     ['baseline_multihop3', 'hop3', 'generations_baseline_multihop3', 'step_by_step_generations_baseline_multihop3']]

# %%


