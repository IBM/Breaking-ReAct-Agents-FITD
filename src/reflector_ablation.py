import os
import json
import argparse
from tqdm import tqdm
from prompts.thought_analysis_prompts import REFLECTOR_USR_PROMPT
from utils import read_all_json_formats
from models import Bam_API_Model
from reflector import ReflectorRunner, get_thought_from_output, ReflectorAnalyzeResults

def run_reflector_ablation(outputs, file_name: str, analysis_dir_path: str, base_model: str = "mistralai/mixtral-8x7b-instruct-v01"):
    """
    Run the reflector with multiple rephrased prompts and save analysis results.

    Args:
        outputs (list): List of outputs to analyze.
        file_name (str): Name of the results file.
        analysis_dir_path (str): Directory path to save analysis results.
        base_model (str): Base model to use for the reflector. Default is Mixtral-8x7b.
    """
    os.makedirs(analysis_dir_path, exist_ok=True)

    with open('./src/prompts/rephrased_reflector_prompts.json', 'r') as f:
        rephrased_prompts = json.load(f)

    hesitation_prompt_format = rephrased_prompts['Hesitation Reflection']['format']
    safe_prompt_format = rephrased_prompts['Safe Reflection']['format']
    hesitation_r_prompts = rephrased_prompts['Hesitation Reflection']['values']
    safe_r_prompts = rephrased_prompts['Safe Reflection']['values']

    reflector_runner = ReflectorRunner(model_name=base_model)

    for prompt_num, hesitation_prompt_fill in hesitation_r_prompts.items():
        safe_prompt_fill = safe_r_prompts.get(prompt_num, None)
        # create the current prompts:
        safe_prompt = safe_prompt_format.replace('<PLACEHOLDER_PROMPT>', safe_prompt_fill)
        hesitation_prompt = hesitation_prompt_format.replace('<PLACEHOLDER_PROMPT>', hesitation_prompt_fill)

        analysis_path = os.path.join(analysis_dir_path, f"analysis_{prompt_num}_{file_name}")

        with open(analysis_path, 'w') as outfile:
            for item in tqdm(outputs, desc=f"Processing prompt {prompt_num}"):
                crnt_item = {
                    'output1': item.get('output1'),
                    'inspected_thought': get_thought_from_output(item.get('output1')),
                    'attack_status': ReflectorRunner.get_attack_status(item, scenario="dh"),
                }

                crnt_item['reflector_safe'] = ReflectorRunner.call_reflector(
                    crnt_item['inspected_thought'], reflector_runner.bam_model,
                    sys_prompt=safe_prompt, usr_prompt=REFLECTOR_USR_PROMPT
                )
                crnt_item['hesitation'] = ReflectorRunner.call_reflector(
                    crnt_item['inspected_thought'], reflector_runner.bam_model,
                    sys_prompt=hesitation_prompt, usr_prompt=REFLECTOR_USR_PROMPT
                )

                print(f"--Hesitation: {crnt_item['hesitation']}, --Safe: {crnt_item['reflector_safe']}")
                outfile.write(json.dumps(crnt_item) + '\n')
                # Calculate and save ASR values after analysis
        analyzer = ReflectorAnalyzeResults(analysis_path=analysis_path)
        old_asr, after_hes_asr, after_sf_asr = analyzer.get_all_asr_values()
        asr_summary_path = os.path.join(analysis_dir_path, f"asr_summary_{prompt_num}_{file_name}.json")

        with open(asr_summary_path, 'w') as summary_file:
            asr_summary = {
                "old_asr": old_asr,
                "after_hesitation_asr": after_hes_asr,
                "after_safe_reflector_asr": after_sf_asr
            }
            json.dump(asr_summary, summary_file, indent=4)
            print(f"Saved ASR summary for prompt {prompt_num} at {asr_summary_path}")


def run_ref_ablation():
    debug = True
    if not debug:
        parser = argparse.ArgumentParser(description="Run reflector ablation with rephrased prompts.")
        parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory or file.")
        parser.add_argument("--results_file_name", type=str, required=True, help="Name of the results file.")
        args = parser.parse_args()

        outputs_path = os.path.join(args.results_dir, args.results_file_name)
        analysis_dir_path = './results/thoughts_analysis_path/'
    else:
        outputs_dir_path = './results/check_published_code/prompted_BAM_API_mistralai/mixtral-8x7b-instruct-v01_InjecAgent/'
        results_file_name = 'test_cases_dh_base_padding:CalculatorCalculate.json'
        outputs_path = os.path.join(outputs_dir_path, results_file_name)
        analysis_dir_path = './results/thoughts_analysis_path_ablation_new/'

    print(f"Running reflector ablation with outputs: {outputs_path} and analysis path: {analysis_dir_path}")
    outputs = read_all_json_formats(outputs_path)
    run_reflector_ablation(outputs=outputs, file_name=results_file_name, analysis_dir_path=analysis_dir_path)


if __name__ == "__main__":
    run_ref_ablation()
