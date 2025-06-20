import json




if __name__ == "__main__":
    session = {}
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                item = json.loads(line)
                chat_id = item['chat_id']
                if chat_id not in session:
                    session[chat_id] = [item]
                else:
                    session[chat_id].append(item)

    print(f"session len: {len(session)}")
    for chat in session:
        print(f" ------------- one session starts --------------- ")
        for req in chat:
            print(f"req: {str(req)}")