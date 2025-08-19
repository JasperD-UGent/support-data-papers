import argparse
import codecs
import os
import sys
import torch
import transformers
from transformers import Pipeline
from typing import Dict, List


def load_llm(name_llm: str) -> Pipeline:
    """Load a given large language model (LLM).
    :param name_llm: Name of the LLM.
    :return: The loaded LLM in the form of a Pipeline object.
    """
    if name_llm == "EuroLLM":
        model_id = "utter-project/EuroLLM-9B-Instruct"

    if name_llm == "Gemma-3-1b":
        model_id = "google/gemma-3-1b-it"

    if name_llm == "Gemma-3-4b":
        model_id = "google/gemma-3-4b-it"

    if name_llm == "Llama-3.1-8B":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    if name_llm == "Llama-3.2-3B":
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

    if name_llm == "Mistral":
        model_id = "mistralai/Ministral-8B-Instruct-2410"

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        device="cuda",
        torch_dtype=torch.bfloat16
    )

    return pipe
    

def load_target_item_set(domain: str, *, target_item_type: str = "BWS-tuples") -> List:
    """Load the set of target items to be annotated by the LLM. The data should be stored in a subdirectory of the 
    'input' folder (with `target_item_type` as the directory name). Each set should be stored in a separate TXT file 
    (with '[`domain`].txt' as the filename). Each line in the TXT should represent a different target item.
    :param domain: Domain for which the target items should be loaded.
    :param target_item_type: Type of the target item set. Defaults to 'BWS-tuples' (i.e. each target item = a 4-tuple 
        to be annotated by means of a best-worst scaling procedure).
    :return: A list containing the target items.
    """
    with codecs.open(os.path.join("input", target_item_type, f"{domain}.txt"), "r", "utf-8") as f:
        l_target_items = [line.strip() for line in f.readlines()]
    f.close()

    return l_target_items


def define_prompt(name_llm: str, domain: str, item: str, *, target_item_type: str = "BWS-tuples") -> List:
    """Define the prompt to be given to the LLM.
    :param name_llm: Name of the LLM.
    :param domain: Domain for which the target items were loaded.
    :param item: Target item to be annotated.
    :param target_item_type: Type of the target item set. Defaults to 'BWS-tuples'. Currently, no other input types 
        are supported. In future experiments, other types might be added.
    :return: The prompt to be given to the model, in the form of a list containing dictionaries defining the system 
        and user roles.
    """
    if target_item_type not in ["BWS-tuples"]:
        raise ValueError(f"`target_item_type` should be one of the following: {['BWS-tuples']}.")
    
    if target_item_type == "BWS-tuples":
        system_prompt = (f"You are an expert in designing courses of Spanish as a foreign language for native speakers of Dutch. "
                         f"You are currently preparing a vocabulary class for a group of advanced learners on {domain} as the specific topic. "
                         f"More specifically, you want to create a vocabulary list in which words are ranked based on how typical they are of the domain of {domain}.")
        user_prompt = (f"Here is set of four Spanish words and their translations to Dutch:\n\n"
                       f"{item}\n\n"
                       f"Give the 'word_ID' of the **most typical** word of the domain of {domain} and the 'word_ID' of the word that is **least typical** of the domain of {domain}. "
                       f"In case a word has multiple meanings, base your judgement on the meaning that has the strongest relation with the domain of {domain} and disregard all other meanings. "
                       f"Respond two word IDs, separated by means of a semicolon: first the 'word_ID' of the word that is **most typical** of the domain of {domain} and then the 'word_ID' of the word that is **least typical** of the domain of {domain}. "
                       f"Do not include any introduction, summary, or explanation.")
    
    if name_llm in ["EuroLLM", "Llama-3.1-8B", "Llama-3.2-3B", "Mistral"]:
        l_msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    if name_llm in ["Gemma-3-1b", "Gemma-3-4b"]:
        l_msgs = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                  {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

    return l_msgs


def prompt_llm(pipe_llm: Pipeline, l_msgs: List, *, target_item_type: str = "BWS-tuples") -> Dict:
    """Prompt the LLM and convert the raw output into a structured dictionary format.
    :param pipe_LLM: Pipeline object holding the LLM.
    :param l_msgs: Prompt to be given to the LLM.
    :param target_item_type: Type of the target item set. Defaults to 'BWS-tuples'.
    :return: A dictionary containing the processed annotation.
    """
    outputs = pipe_llm(
        l_msgs,
        max_new_tokens=16
    )
    annot_raw = outputs[0]["generated_text"][-1]["content"].strip()

    if target_item_type == "BWS-tuples":
        annot_no_spaces = annot_raw.replace(" ", "")
        best = None
        worst = None

        if (len(annot_no_spaces) >= 3 
                and annot_no_spaces[0].isdigit() 
                and annot_no_spaces[2].isdigit() 
                and annot_no_spaces[0] != annot_no_spaces[2]):
            
            if int(annot_no_spaces[0]) in [1, 2, 3, 4] and int(annot_no_spaces[2]) in [1, 2, 3, 4]:
                best = int(annot_no_spaces[0])
                worst = int(annot_no_spaces[2])

        d_annot = {"best": best, "worst": worst, "raw": annot_raw}

    return d_annot


def main(name_llm: str) -> None:
    """Entry point for the script. Loads the LLM, submits the prompt, processes the LLM output, and writes the 
    results to a TXT.
    :param name_llm: Name of the LLM.
    :return: `None`
    """
    # create output directory if not exists
    direc_outp = "output"

    if not os.path.isdir(os.path.join(direc_outp)):
        os.makedirs(os.path.join(direc_outp))

    # load the LLM
    pipe_llm = load_llm(name_llm)

    # loop over the four domains
    for domain in ["economics", "health", "law", "migration"]:

        # load the target item set
        l_target_items = load_target_item_set(domain)

        # loop over the items in the set
        for item in l_target_items:

            # define the prompt for the item
            l_messages = define_prompt(name_llm, domain, item)

            # submit the prompt to the LLM and process its output
            d_annot = prompt_llm(pipe_llm, l_messages)

            # append the processed LLM output to a TXT (each line corresponds to target item and LLM output 
            # separated by a tab)
            with codecs.open(os.path.join("output", f"{domain}_{name_llm}.txt"), "a", "utf-8") as f_write:
                f_write.write("\t".join([item, str(d_annot)]) + "\n")
            f_write.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "name_llm",
        choices=["EuroLLM", "Gemma-3-1b", "Gemma-3-4b", "Llama-3.1-8B", "Llama-3.2-3B", "Mistral"], type=str,
        help="Name of the generative LLM that should be used to rank the target items."
    )
    main(**vars(parser.parse_args()))
