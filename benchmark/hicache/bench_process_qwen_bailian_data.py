import json




if __name__ == "__main__":
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"type of f = {type(data)}")