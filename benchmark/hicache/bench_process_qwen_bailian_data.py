import json

if __name__ == "__main__":
    chat_dict = {}
    parent = {}
    with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                item = json.loads(line)
                print(f"{str(item)}")
                #chat_id = item['chat_id']
                #chat_dict[chat_id] = item
                #parent_chat_id = item['parent_chat_id']

    # link_count = 0
    # for chat_id, item in chat_dict.items():
    #     parent_chat_id = item['parent_chat_id']
    #     if parent_chat_id != -1:
    #         link_count += 1
    #         chat_dict[parent_chat_id]['child_id'] = chat_id
    #         print(f"link, from {parent_chat_id} to {chat_id}")

    # print(f"link count = {link_count}")

    # print(f"chat_dict len: {len(chat_dict)}")
    # for chat_id, item in chat_dict.items():
    #     req_count = 0
    #     if item['parent_chat_id'] != -1:
    #         continue
    #     print(f" ------------- one session starts --------------- ")
    #     while True:
    #         req_count += 1
    #         print(f"req: {str(item)}")
    #         if 'child_id' in item:
    #             item = chat_dict[item['child_id']]
    #         else:
    #             break
    #     print(f" ------------- one session ends, req count = {req_count} --------------- ")