import pickle
import os
import sys
import pickle
import openai
from openai import OpenAI
import pandas
from pydantic import BaseModel

def build_client():
    KEY = 'sk-w7weiFTL6mzJrr3P4S3wEA'
    client = openai.OpenAI(api_key=KEY, base_url="https://litellm-proxy-153298433405.us-east1.run.app/")
    return client



class RustSafety(BaseModel):
    safety: str
    poc: str
    explain: str


def build_request(client: OpenAI, text: str,  model: str):
    print('='*20)



    messages = [
        {
            "role": "system",
            "content": "You are an expert in Rust language and memory safety, your task is to judge the given function, starting with `>` is safe or unsafe in Rust."
                       "Whether it's safe or unsafe depends on if we can build a PoC code using safe Rust(we can call the error functions) to trigger unsound issues."
                       "If it's actually unsafe, you need to provide a PoC code using safe Rust to trigger unsound issue."
                       'Please reply in the following json format for me to parse: {"safety": "safe or unsafe", "PoC": "optional PoC code", "explain": "description"}'

        },
        {
            "role": "user",
            "content": f"The following function with context is:\n```rust\n{text}\n```\n"
        }
    ]
    # print(messages[0]['content'])

    completion = client.beta.chat.completions.parse(
        model=model,
        messages= messages,
        response_format=RustSafety
    )
    obj:  RustSafety= completion.choices[0].message.parsed
    # print(obj)

    messages.append({"role":"assistant", "content": obj})
    return (messages, obj)

#108 339
def parse_case(row: dict):
    ls = row['function_text'].splitlines(keepends=True)
    if len(ls) < 2:
        return False
    for l in ls:
        if '#[doc = r"Writes raw bits to the field"]' in l or 'pub fn bits(&self) -> u32 {' in l:
            return False
    is_early = False
    for l in ls:
        if '>' in l and ('set_len' in l or 'set_size' in l or ' unsafe ' in l):
            is_early = True
            break
    return True


if __name__ == '__main__':
    pk = sys.argv[1]
    client = build_client()
    ct = 0
    with open(pk, 'rb') as fp:
        data = pickle.load(fp)
        nd = []
        for d in data:
            if parse_case(d):
                msgs, obj = build_request(client, d['function_text'], 'gpt-4o')
                if obj.safety.lower() == 'unsafe':
                    print('-'*20)
                    print(d['function_text'])
                    print(d['file_location'])
                    print(obj.poc)
                    print(obj.explain)
                    print('$'*20)
                    ct += 1
        print(ct, len(data))

