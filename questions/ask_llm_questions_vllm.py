import pandas as pd
import ast
from vllm import LLM, SamplingParams
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import random
import torch
import re
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from datasets import Dataset
from openai import OpenAI
import os
import gc
import argparse

parser = argparse.ArgumentParser(
                    prog= 'RunQuestions',
                    description= 'Generates multiple choice questions.')

#Sentence type
parser.add_argument('--type', type= str, choices = ["open", "open_cot", "mcq_2", "mcq_4", "mcq_seojin", "open_source",
                                                   "open_source_stage_2", "baseline_mapping", "antonym_mapping", "pseudoword_mapping", "together_baseline", "v_source", "n_source"])

args = parser.parse_args()

client = OpenAI()

deepseek_api_key = os.environ['DEEPSEEK_API_KEY']

deepseek_client = OpenAI(api_key= deepseek_api_key, base_url= "https://api.deepseek.com")

def extract_response(response):
    #should be in format /boxed[A]/
    match = re.findall(r'(/boxed\[|/boxed |/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)(.*?)(}|>|\]|}\\|>\\|\]\\|\))', response)
    try:
        return match[-1][1].upper()
    except:
        sliced = response
        if "</think>" in sliced: #for reasoning models
            sliced = sliced[sliced.find("</think>") + len("</think>"):]
        try:
            match = re.findall(r'(/boxed\[|/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)(.*?)(}|>|\]|}\\|>\\|\]\\)', response)
            return match[-1][1].upper()
        except:
            if("boxed" in response):
                sliced = sliced[sliced.find("boxed"):]
            elif "answer:" in response:
                sliced = sliced[sliced.find("answer:") + len("answer:"):]
            elif "final answer" in response.lower():
                sliced = sliced[sliced.find("final answer") + len("final answer"):]
            elif "Answer:" in sliced:
                sliced = sliced[sliced.find("Answer:") + len("Answer:"):]
            elif "ANSWER:" in sliced:
                sliced = sliced[sliced.find("ANSWER:") + len("ANSWER:")]
            elif "Final Answer:" in sliced:
                sliced = sliced[sliced.find("Final Answer:") + len("Final Answer:"):]
            elif "Final answer" in sliced:
                sliced = sliced[sliced.find("Final answer") + len("Final answer"):]
            elif "**Answer**" in response:
                sliced = sliced[sliced.find("**Answer**") + len("**Answer**"):]
            elif "*Answer*" in response:
                sliced = sliced[sliced.find("*Answer*") + len("*Answer*"):]
            else:
                return "E"
            if("A" in sliced):
                sliced = "A"
            elif("B" in sliced):
                sliced = "B"
            elif("C" in sliced):
                sliced = "C"
            elif("D" in sliced):
                sliced = "D"
            return sliced

def rewrite_prompt(prompt_file: str, prompt: str, model_name: str, quant: int):
    device = "cuda"

    if(quant == 4):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_use_double_quant= True,
        )
    elif(quant == 8):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        return

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, quantization_config= bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    master_prompt = "Please rewrite the following PROMPT so that you can more accurately understand the question.\nPROMPT:" + '"' + prompt + '"'
    with open(prompt_file, "w+") as f:
        if("-it" in model_name): #for gemma instruct models
            chat = [
                { "role": "user", "content": master_prompt}
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize= False, add_generation_prompt=True)
            inputs = tokenizer.encode(prompt, add_special_tokens= False, return_tensors= "pt")
            inputs = inputs.to(device)
            outputs = model.generate(input_ids= inputs, max_length= 1000, do_sample= False, temperature= None, top_p= None)
            text = tokenizer.batch_decode(outputs)[0]
            f.write(text)
        else:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="cuda",
            )
            messages = [
                        {"role": "user", "content": master_prompt},
                    ]
            outputs = pipe(
                messages,
                max_new_tokens= 1000,
                do_sample=False,
                temperature= None,
                top_p= None
            )
            text = outputs[0]["generated_text"][-1]['content']
            f.write(text)

def clean_input(text, is_tar):
    if(is_tar):
        text = str(text)
        quoted_word = text[text.rfind("QUESTION"):]
        to_question = text[:text.rfind("QUESTION")]
        quoted_word = quoted_word[quoted_word.find('"') + 1:]
        quoted_word = quoted_word[:quoted_word.find('"')]
        options = text[text.rfind(":"):]
        if(quoted_word == ""):
            print("Issue!\n")
        text = re.sub(r"':", ":", text)
        text = to_question + 'QUESTION: Based on the given text, which word is being described or explained by the metaphorical word "' + quoted_word + '"? Choose the best option from the following' + options

    text = text.replace("Choose the best word from the options below:", "Choose the best option from the following:")

    # Replace 2 or more consecutive quotes with a single quote
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r"'{2,}", "'", text)

    # Replace colon followed by a quote with just colon
    text = re.sub(r':"', ':', text)
    text = re.sub(r"':", ":", text)

    #strip leading and trailing single and double quotes
    text = re.sub(r"^['\"]|['\"]$", "", text)

    #strip any quotes before the word "TEXT"
    text = re.sub(r'[""\'](?=TEXT)', '', text)
    text = re.sub(r'["\'](?=TEXT)', '', text)

    text = re.sub(r'\n"\n', '\n', text)

    return text

def finalize_question(input: str, is_open):
    if(is_open):
        return input + "FINAL ANSWER FORMATTING INSTRUCTION: Your final answer MUST be a single word (if possible), in the form '/boxed[word]', at the end of your response."
    return input + "\nFINAL ANSWER FORMATTING INSTRUCTION: Your final answer MUST be a single letter, in the form '/boxed[letter]', at the end of your response."

def get_gpt_output(input: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content":
                [
                    {"type": "text", "text": input},
                ],
            }
        ],
        temperature= 0,
        presence_penalty= 0.0,
        max_tokens= 1000,
        frequency_penalty= 0.0,
        top_p= 1.0,
        seed= 42
    )
    return response.choices[0].message.content

def get_gpt_reasoning_output(input: str):
    response = client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "user", "content":
                [
                    {"type": "text", "text": input},
                ],
            }
        ],
        #temperature= 0,
        presence_penalty= 0.0,
        max_completion_tokens= 1000,
        frequency_penalty= 0.0,
        reasoning_effort= "high",
        top_p= 1.0,
        seed= 42
    )
    final = response.choices[0].message.content
    return final

def get_deepseek_reasoning_output(input):
    try:
        prompt = "" if input is None else str(input)
    except Exception:
        print(f"Invalid Input: {input}")
        return "invalid deepseek response"
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature= 0,
            presence_penalty= 0.0,
            max_completion_tokens= 1000,
            frequency_penalty= 0.0,
            reasoning_effort= "high",
            top_p= 1.0,
            seed= 42
        )
        return response.choices[0].message.content or ""
    except Exception:
        print(f"Invalid Input: {input}")
        return "invalid deepseek response"

def permute(original, permutation, text: str):
    original = ast.literal_eval(original)
    permutation = ast.literal_eval(permutation)
    if(text != "E"):
        try:
            unshuffled_answer = original[permutation[text]]
        except:
            unshuffled_answer = "E"
    else:
        unshuffled_answer = "E"
    return unshuffled_answer

def get_other_output(tokenizer, model, message: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def get_vllm_outputs_batched(llm: LLM, questions: list, model_name: str, enable_thinking: bool, max_tokens: int = 1000) -> list:
    tokenizer = llm.get_tokenizer()
    is_qwen = "qwen" in model_name.lower()
    is_gemma = "gemma" in model_name.lower()
    if is_qwen or is_gemma:
        max_tokens = 8192
    thinking_kwargs = {"enable_thinking": enable_thinking} if is_qwen or is_gemma else {}
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            **thinking_kwargs,
        )
        for q in questions
    ]
    sampling_params = SamplingParams(temperature= 0, top_p= 1.0, max_tokens= max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

def test_model(model_name: str, nickname: str, questions_file: str, is_open: bool, enable_thinking: bool = True):
    print(f"\nLoading questions from: {questions_file}")
    questions_df = pd.read_csv(questions_file, encoding="utf-8", engine="python", on_bad_lines="warn")

    print(f"Loaded {len(questions_df)} questions")
    print(questions_df.head())

    is_api = "gpt" in model_name or "deepseek-R1" in model_name
    is_fp8 = "FP8" in model_name or "fp8" in model_name
    is_gemma_big = "gemma-4-31" in model_name.lower() or "gemma-4-26" in model_name.lower()
    tp_size = 1 if any(tag in model_name for tag in ["3.5-4B", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b"]) else 2
    #tp_size = 2

    if not is_api:
        print(f"Loading model: {model_name}")
        is_qwen = "qwen" in model_name.lower()
        gpu_util = 0.9 if is_qwen or is_gemma_big else 0.8
        if is_fp8:
            # Pre-quantized FP8 checkpoint — quantization auto-detected from model config.
            # On A100 PCIE (compute capability 8.0) vLLM falls back to W8A16.
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                trust_remote_code=True,
                enforce_eager=True,
                gpu_memory_utilization=gpu_util,
            )
        elif(is_gemma_big):
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                dtype="bfloat16",
                trust_remote_code=True,
                enforce_eager=True,
                gpu_memory_utilization=gpu_util,
                limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                #**extra_kwargs,
            )
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                dtype="bfloat16",
                trust_remote_code=True,
                enforce_eager=True,
                gpu_memory_utilization=gpu_util,
                #**extra_kwargs,
            )
        print(f"Model loaded")

    with torch.no_grad():
        dataset = Dataset.from_pandas(questions_df)
        print("Dataset converted from pandas")

        def debug_map_fn(x, output):
            print(f"\nQuestion: {x['normal_question']}")
            print(f"Model output: {output}")
            return {"normal_question": x["normal_question"], "full_answer": output}

        rows = []
        if not is_api:
            questions = [x["normal_question"] for x in dataset]
            answers = get_vllm_outputs_batched(llm, questions, model_name=model_name, enable_thinking=enable_thinking, max_tokens=1000)
            for x, answer in zip(dataset, answers):
                row = {"normal_question": x["normal_question"], "full_answer": answer}
                if not is_open:
                    row["original"] = x["original"]
                    row["permutation"] = x["permutation"]
                rows.append(row)
            dataset = Dataset.from_list(rows)
        elif "gpt-4o" in model_name:
            dataset = dataset.map(lambda x: debug_map_fn(x, get_gpt_output(x["normal_question"])))
        elif "gpt-o3" in model_name:
            dataset = dataset.map(lambda x: debug_map_fn(x, get_gpt_reasoning_output(x["normal_question"])))
        elif "deepseek-R1" in model_name:
            dataset = dataset.map(lambda x: debug_map_fn(x, get_deepseek_reasoning_output(x["normal_question"])))

        print("\nApplying post-processing steps")
        if(is_open):
            dataset = dataset.map(lambda x: {**x, "shuffled_answer": extract_response(x["full_answer"])})
        else:
            dataset = dataset.map(lambda x: {**x, "shuffled_answer": extract_response(x["full_answer"])})
            dataset = dataset.map(lambda x: {**x, "answer": permute(x["original"], x["permutation"], x["shuffled_answer"])})

        print(f"\nSaving results to: results/{nickname}.csv")
        dataset.to_csv("results/" + nickname + ".csv")

        if not is_api:
            print(f"Cleaning up model: {model_name}")
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def quote_text_block(prompt: str) -> str:
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("TEXT:"):
            # Find where "TEXT:" ends and take the rest of the line
            prefix, content = line.split("TEXT:", 1)
            lines[i] = f"{prefix}TEXT: \"{content}\""
            break  # only first TEXT block
    return "\n".join(lines)

def remove_blank_after_question(prompt: str) -> str:
    lines = prompt.splitlines()
    new_lines = []
    skip_blank = False

    for line in lines:
        if skip_blank:
            # skip lines that are empty or only whitespace
            if line.strip() == "":
                continue
            else:
                skip_blank = False  # stop skipping once we reach a real line

        new_lines.append(line)

        # if this line starts with QUESTION:, skip following blank lines
        if line.lstrip().startswith("QUESTION:"):
            skip_blank = True

    return "\n".join(new_lines)

def fix_open_prompt(prompt: str) -> str:
    """
    Clean and fix open-ended prompts:
    1. Wrap the TEXT: block in quotes if not already quoted.
    2. Remove extra blank lines immediately after TEXT:.
    3. Remove leading spaces from TEXT:, INSTRUCTIONS:, FINAL ANSWER lines.
    """
    lines = prompt.splitlines()
    new_lines = []
    skip_blank_after_text = False

    # Lines that we want left-aligned
    left_align_prefixes = ["TEXT:", "INSTRUCTIONS:", "FINAL ANSWER FORMATTING INSTRUCTION:", "EXAMPLE"]

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Left-align certain headers
        for prefix in left_align_prefixes:
            if stripped.startswith(prefix):
                line = stripped  # remove all leading spaces

        # Step 1: Quote TEXT block if not already quoted
        if stripped.startswith("TEXT:"):
            prefix, content = line.split("TEXT:", 1)
            content = content.strip()
            if not (content.startswith('"') and content.endswith('"')):
                line = f"TEXT: \"{content}\""
            skip_blank_after_text = True  # remove blank lines after TEXT

        # Step 2: Remove blank lines immediately after TEXT:
        elif skip_blank_after_text:
            if stripped == "":
                continue
            else:
                skip_blank_after_text = False

        new_lines.append(line)

    return "\n".join(new_lines)

def construct_mcq(prompt: str, a: str, b: str, c: str, d: str, is_2: bool, is_seojin: bool):
    prompt = quote_text_block(prompt)
    while "\n\n" in prompt:
        prompt = prompt.replace("\n\n", "\n")
    prompt = re.sub(r"^\s+QUESTION:", "QUESTION:", prompt, flags= re.MULTILINE)
    prompt = remove_blank_after_question(prompt)
    if(is_2):
        if(is_seojin):
            options = {0: "A", 1: "B", 2: "C", 3: "D"}
            important_list = [a, b, c, d]

            original = {}
            for answer in important_list:
                original[answer] = options[important_list.index(answer)]

            random.shuffle(important_list)

            a_idx = -1
            b_idx = -1
            i = 0
            for answer in important_list:
                if(answer == a):
                    a_idx = i
                elif(answer == b):
                    b_idx = i
                i += 1

            a_letter = options[a_idx]
            b_letter = options[b_idx]

            exclude = {a_idx, b_idx}

            while True:
                n = random.randint(0, 3)
                if n not in exclude:
                    break

            exclude = {a_idx, b_idx, n}

            while True:
                m = random.randint(0, 3)
                if m not in exclude:
                    break

            important_list[n] = f"Both options {a_letter} and {b_letter}"
            important_list[m] = "None of the options"

            permutation = {}
            for answer in important_list:
                permutation[answer] = options[important_list.index(answer)]
            permutation = {v: k for k,v in permutation.items()}

            final = prompt + "\nA) " + important_list[0] + "\nB) " + important_list[1] + "\nC) " + important_list[2] + "\nD) " + important_list[3]
            final = finalize_question(final, False)
            return final, original, permutation
        else:
            options = {0: "A", 1: "B"}
            important_list = [a, b]

            original = {}
            for answer in important_list:
                original[answer] = options[important_list.index(answer)]

            random.shuffle(important_list)

            permutation = {}
            for answer in important_list:
                permutation[answer] = options[important_list.index(answer)]
            permutation = {v: k for k,v in permutation.items()}

            final = prompt + "\nA) " + important_list[0] + "\nB) " + important_list[1]
            final = finalize_question(final, False)
            return final, original, permutation
    else:
        options =  {0:"A", 1:"B", 2: "C", 3: "D"}
        important_list = [a, b, c, d]

        original = {}
        for answer in important_list:
            original[answer] = options[important_list.index(answer)]

        random.shuffle(important_list)

        permutation = {}
        for answer in important_list:
            permutation[answer] = options[important_list.index(answer)]
        permutation = {v: k for k,v in permutation.items()}

        final = prompt + "\nA) " + important_list[0] + "\nB) " + important_list[1] + "\nC) " + important_list[2] + "\nD) " + important_list[3]
        final = finalize_question(final, False)
        return final, original, permutation

def create_mcqs() -> None:
    data1 = pd.read_csv("data/for_testing.csv")
    data2 = pd.read_csv("data/for_testing.csv")
    data1[["normal_question", "original", "permutation"]] = data1.apply(lambda x: pd.Series(construct_mcq(x["mcq_prompt"]
    , x["A"], x["B"], x["C"], x["D"], True, False)), axis= 1)
    data2[["normal_question", "original", "permutation"]] = data2.apply(lambda x: pd.Series(construct_mcq(x["mcq_prompt"]
    , x["A"], x["B"], x["C"], x["D"], False, False)), axis= 1)
    data1.to_csv("data/mcq_2.csv", index= False)
    data2.to_csv("data/mcq_4.csv", index= False)

def create_seojin_mcqs() -> None:
    data = pd.read_csv("data/for_testing.csv")
    data[["normal_question", "original", "permutation"]] = data.apply(lambda x: pd.Series(construct_mcq(x["mcq_prompt"]
    , x["A"], x["B"], x["C"], x["D"], True, True)), axis= 1)
    data.to_csv("data/mcq_seojin.csv", index= False)

def create_open(is_cot: bool) -> None:
    data = pd.read_csv("data/for_testing.csv")
    if(is_cot):
        data["normal_question"] = data["open_prompt"].apply(lambda x: "DIRECTIVE: Think step by step.\n" + finalize_question(fix_open_prompt(x), True))
        data.to_csv("data/cot_open_questions.csv", index= False)
    else:
        data["normal_question"] = data["open_prompt"].apply(lambda x: finalize_question(fix_open_prompt(x), True))
        data.to_csv("data/open_questions.csv", index= False)

NEW_MODELS = [
    # Small (vLLM bfloat16)
    ("google/gemma-3-1b-it", "gemma-1b", False),
    ("google/gemma-3-4b-it", "gemma-4b", False),
    ("Qwen/Qwen3.5-4B", "qwen3.5-4b-thinking", True),
    ("Qwen/Qwen3.5-4B", "qwen3.5-4b-nothinking", False),

    # Medium (vLLM bfloat16)
    ("google/gemma-3-12b-it", "gemma-12b", False),
    ("Qwen/Qwen3.5-9B", "qwen3.5-9b-thinking", True),
    ("Qwen/Qwen3.5-9B", "qwen3.5-9b-nothinking", False),

    # Large (vLLM bfloat16 / W8A16 for FP8)
    ("google/gemma-3-27b-it", "gemma-27b", False),
    ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3.5-35b-fp8-thinking", True),
    ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3.5-35b-fp8-nothinking", False),

    # API
    ("gpt-4o", "gpt-4o", False),
    ("deepseek-R1", "deepseek-R1", True),
]

'''Remember to change NEW_GEMMA_ONLY back to NEW_MODELS'''

if __name__ == '__main__':
    #needs fixed maybe
    if(args.type == "open"):
        test_model("gpt-4o", "open_target/gpt-4o-target-open", "data/open_questions.csv", True)
        test_model("deepseek-R1", "open_target/deepseek-R1-target-open", "data/open_questions.csv", True)

        print("results saved to directory: results/open_target")

    elif args.type == "v_source":
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mcq_source/vanilla_source_{short}", "MetaphorMemorizationOrReasoning/source_questions/data/updated_mcq_vanilla_source_questions.csv", True, thinking)
            print("results saved to directory: results/mcq_source")

    elif args.type == "n_source":
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mcq_source/normal_source_{short}", "MetaphorMemorizationOrReasoning/source_questions/data/updated_mcq_normal_source_questions.csv", True, thinking)
            print("results saved to directory: results/mcq_source")

    elif args.type == "baseline_mapping":
        for model_name, short, thinking in NEW_MODELS: #temp edit
            test_model(model_name, f"mapping/final_baseline_original_target_{short}", "mapping_data/for_llms/final_baseline_original_target.csv", True, thinking)
            test_model(model_name, f"mapping/final_baseline_our_target_{short}", "mapping_data/for_llms/final_baseline_our_target.csv", True, thinking)
            print("results saved to directory: results/mapping")

    elif args.type == "antonym_mapping":
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mapping/final_antonym_original_target_{short}", "mapping_data/for_llms/final_antonym_original_target.csv", True, thinking)
            test_model(model_name, f"mapping/final_antonym_our_target_{short}", "mapping_data/for_llms/final_antonym_our_target.csv", True, thinking)
            print("results saved to directory: results/mapping")

    elif args.type == "pseudoword_mapping":
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mapping/final_pseudoword_original_target_{short}", "mapping_data/for_llms/final_pseudoword_original_target.csv", True, thinking)
            test_model(model_name, f"mapping/final_pseudoword_our_target_{short}", "mapping_data/for_llms/final_pseudoword_our_target.csv", True, thinking)
            print("results saved to directory: results/mapping")

    elif(args.type == "open_source"):
        #API
        test_model("gpt-4o", "open_source/gpt-4o-source-open-full", "MetaphorMemorizationOrReasoning/annotations/for_question_generation.csv", True)
        test_model("deepseek-R1", "open_source/deepseek-R1-source-open-full", "MetaphorMemorizationOrReasoning/annotations/for_question_generation.csv", True)
        print("results saved to directory: results/open_source")

    elif(args.type == "open_source_stage_2"):
        print("Second stage open...\n")
        test_model("gpt-4o", "open_source/gpt-4o-source-open-2nd-stage-full", "open_source_data/gpt_2nd_stage_prompts.csv", True)
        test_model("deepseek-R1", "open_source/deepseek-R1-source-open-2nd-stage-full", "open_source_data/deepseek_2nd_stage_prompts.csv", True)
        print("results saved to directory: results/open_source")

    elif(args.type == "mcq_2"):
        print("mcq_2...\n")
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mcq_target/2_option/{short}-target-mcq_2", "data/mcq_2.csv", False, thinking)
            print("results saved to directory: results/mcq_target/2_option")

    elif(args.type == "mcq_seojin"):
        print("mcq_none_or_all...\n")
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mcq_target/4_option/{short}-target-mcq_none_or_all", "data/mcq_seojin.csv", False, thinking)
            print("results saved to directory: results/mcq_target/4_option")

    else:
        print("mcq_4...\n")
        for model_name, short, thinking in NEW_MODELS:
            test_model(model_name, f"mcq_target/4_option/{short}-target-mcq_4", "data/mcq_4.csv", False, thinking)
            print("results saved to directory: results/mcq_target/4_option")
