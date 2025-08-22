import os
import time
from os.path import *
from pathlib import Path
import pandas as pd
import requests
import json
from typing import *
import urllib3
from abc import *
import sacrebleu
from evaluate import load
from statistics import mean
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_PROMPT_POSITIVE_TO_NEGATIVE = 'Convert the following text from positive to negative. Do not write comments or explanations. Just write the text in negative sentiment.'
USER_PROMPT_NEGATIVE_TO_POSITIVE = 'Convert the following text from negative to positive. Do not write comments or explanations. Just write the text in positive sentiment.'

ACCESS_TOKEN = '' # TODO: Add the token

def make_parent_dirs(path: str):
    (Path(path)
     .parent
     .mkdir(parents=True, exist_ok=True))

def read_from_file(file_path: str, default_value=None):
    if not exists(file_path):
        return default_value
    with open(file_path, mode='r') as file:
        return file.read()


def read_from_json_file(file_path: str, default_value=None):
    if not exists(file_path):
        print(f'{file_path} does not exist')
        return default_value
    with open(file_path, mode='r') as json_file:
        return json.load(json_file)


def write_to_file(file_path: str, content: Union[str, List[str]], mode='w'):
    make_parent_dirs(file_path)
    with open(file_path, mode=mode) as file:
        if type(content) is str:
            file.write(content)
        else:
            for line in content:
                file.write(f'{line}\n')


def append_to_file(file_path: str, content: Union[str, List[str]]):
    write_to_file(file_path, content, mode='a')


def write_to_json_file(file_path: str, content):
    make_parent_dirs(file_path)
    with open(file_path, mode='w') as json_file:
        return json.dump(content, json_file, ensure_ascii=False)



def get_chat_completion(messages):
    payload = {
        "messages": messages,
        "temperature": 0.6,
        "stream": False,
        "model": "allam",
        "top_p": 0.98,
        "n": 1,
        "add_generation_prompt": True,
        "echo": False,
        "stop": " </s>",
    }
    response = requests.post(
        'https://vllm-v19.allam.ai/v1/chat/completions',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        },
        data=json.dumps(payload),
        timeout=150,
        verify=False,
    )
    if response.status_code == 200:
        chat_response_data = response.json()
        return chat_response_data['choices'][0]['message']['content']
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return None


def create_prompt_messages(user_prompt, shots, dialect_sentence):
    return [
        *[
            shot_message
            for dialect_text_i, msa_text_i in shots
            for shot_message in [
                {
                    'role': 'user',
                    'content': f'{user_prompt}\nText: {dialect_text_i}',
                },
                {
                    'role': 'assistant',
                    'content': f'{msa_text_i}',
                },
            ]
        ],
        {
            'role': 'user',
            'content': f'{user_prompt}\nText: {dialect_sentence}',
        },
    ]


def convert_sentiment(sentence, polarity, shots=None):
    if shots is None:
        shots = []
    prompt_messages = create_prompt_messages(
        user_prompt=USER_PROMPT_NEGATIVE_TO_POSITIVE if polarity == -1 else USER_PROMPT_POSITIVE_TO_NEGATIVE,
        shots=shots,
        dialect_sentence=sentence,
    )
    model_output = None
    sleep_period = 1.0
    while model_output is None:
        model_output = get_chat_completion(prompt_messages)
        if model_output is not None:
            break
        print(f"Sleep for {sleep_period} and then retry")
        time.sleep(sleep_period)
        sleep_period *= 2
    model_output = model_output.strip()
    if '\n' in model_output:
        model_output = model_output.split('\n')[0]
    return model_output

output_files = []
i = 0
for shots_count in [0, 1, 5, 10, 20]:
    for polarity in [-1, +1]:
        dataframe = pd.read_csv(f'./data/SentimentTransfer.csv')
        dataframe = dataframe[dataframe['polarity'] == polarity]

        shots_dataframe = dataframe[dataframe['split'] == 'train'][:shots_count]
        shots = list(zip(
            shots_dataframe['source'].tolist(),
            shots_dataframe['target'].tolist(),
        ))

        test_dataframe = dataframe[dataframe['split'] == 'test']
        source_sentences = test_dataframe['source'].tolist()
        target_sentences = test_dataframe['target'].tolist()
        output_file = f'./output/{i}.json'
        i += 1
        output_files.append(output_file)
        write_to_json_file(
            output_file,
            content={
                'polarity': 'neg-2-pos' if polarity == -1 else 'pos-2-neg',
                'shots_count': shots_count,
                'shots': shots,
                'source': source_sentences,
                'actual_target': target_sentences,
                'predicted_target': [
                    convert_sentiment(
                        sentence=sentence,
                        polarity=polarity,
                        shots=shots,
                    )
                    for sentence in tqdm(source_sentences)
                ],
            }
        )


class TGMetric(ABC):

    @abstractmethod
    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        pass


class BleuMetric(TGMetric):

    def __init__(self):
        self.bleu = sacrebleu.BLEU()

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str]):
        bleu_score = self.bleu.corpus_score(predictions, [ground_truth]).score
        return {
            'bleu': bleu_score,
        }
    
    
class ChrfMetric(TGMetric):

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str], word_order=2):
        chrf = sacrebleu.CHRF(word_order=word_order)
        chrf_score = chrf.corpus_score(predictions, [ground_truth]).score
        return {
            'chrf': chrf_score,
        }


class CometMetric(TGMetric):

    def __init__(self):
        self._comet_metric = load('comet')

    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        compute_result = self._comet_metric.compute(
            predictions=predictions,
            references=ground_truth,
            sources=sources,
            progress_bar=True,
        )
        return {'comet': compute_result['mean_score']}


class BertScoreMetric(TGMetric):

    def __init__(self, lang):
        self._bertscore = load("bertscore")
        self._lang = lang

    def __call__(self, ground_truth: List[str], predictions: List[str], sources: List[str],) -> Dict[str, float]:
        assert len(ground_truth) == len(predictions)
        compute_result = self._bertscore.compute(
            predictions=predictions,
            references=ground_truth,
            lang=self._lang,
            verbose=True,
        )
        return {
            'bertscore_precision': mean(compute_result['precision']),
            'bertscore_recall': mean(compute_result['recall']),
            'bertscore_f1': mean(compute_result['f1']),
        }


metrics = [
    BleuMetric(),
    CometMetric(),
    BertScoreMetric(lang='ar'),
    ChrfMetric(),
]

evaluation_records = []
for output_file in output_files:
    data = read_from_json_file(output_file)
    evaluation_records.append(
        {
            'polarity': data['polarity'],
            'shots_count': data['shots_count'],
            **{
                k: v
                for metric in metrics
                for k, v in metric(
                    ground_truth=data['actual_target'],
                    predictions=data['predicted_target'],
                    sources=data['source'],
                ).items()
            },
        }
    )

evaluation_dataframe = pd.DataFrame.from_records(evaluation_records)
evaluation_dataframe.to_csv('./output/evaluation.csv', index=False)
