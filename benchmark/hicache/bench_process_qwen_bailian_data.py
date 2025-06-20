import json




if __name__ == "__main__":
    session = {}
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                chat_id = json.loads(line['chat_id'])
                if chat_id not in session:
                    session[chat_id] = [line]
                else:
                    session[chat_id].append(line)

    print(f"session len: {len(session)}")
    for chat in session:
        print(f" ------------- one session starts --------------- ")
        for req in chat:
            print(f"req: {str(req)}")