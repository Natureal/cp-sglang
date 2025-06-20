import json




if __name__ == "__main__":
    data_list = []
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data_list.append(json.loads(line))

    print(f"line1 = {str(data_list[0])}")
    print(f"line2 = {str(data_list[1])}")
    print(f"line3 = {str(data_list[1])}")