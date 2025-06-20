import json




if __name__ == "__main__":
    chat_dict = {}
    parent = {}
    session = {}
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                item = json.loads(line)
                chat_id = item['chat_id']
                chat_dict[chat_id] = item
                parent_chat_id = item['parent_chat_id']
                if parent_chat_id == -1:
                    if chat_id not in session:
                        session[chat_id] = [item]
                    else:
                        print(f"error happened, chat id replicates {chat_id}")

    for chat_id, item in chat_dict.items():
        parent_chat_id = item['parent_chat_id']
        if parent_chat_id != -1:
            chat_dict[parent_chat_id]['child_id'] = chat_id

    print(f"session len: {len(session)}")
    for chat_id, item in session.items():
        print(f" ------------- one session starts --------------- ")
        while True:
            print(f"req: {str(item)}")
            if 'child_id' in item:
                item = chat_dict[item['child_id']]
            else:
                break