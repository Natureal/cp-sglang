import argparse
import asyncio
import json
import queue
import random
import pickle
import threading
import random
import time
import os
from datetime import datetime
from typing import Optional
from collections import deque

import aiohttp
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)

#request_rate_list = [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
request_rate_list = [0.5, 0.2, 0.1, 0.05]
request_rate_map = {}
client_id_to_idx = {}
idx_to_client_id = {}
request_history = []

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=256,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=128,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    return parser.parse_args()


async def async_request_sglang_generate(
    payload,
    url,
    pbar: Optional[tqdm] = None,
):
    """
    Sends a streaming request to the server. Gathers text token-by-token.
    """
    async with aiohttp.ClientSession() as session:
        headers = {}
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


def gen_payload(prompt, output_len):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "lora_path": "",
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    return payload


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl"):
    """Append the data with a timestamp to the specified JSONL file."""
    timestamped_data = {"timestamp": datetime.now().isoformat(), **data}
    try:
        with open(file_path, "a") as file:
            file.write(
                json.dumps(timestamped_data) + "\n"
            )  # Write as a single line in JSONL format
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        # Construct the base URL for requests
        self.url = f"http://{args.host}:{args.port}/generate"

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.start_time = None
        self.finished_time = None
        self.num_clients = args.num_clients

        self.sent_requests = 0
        self.completed_requests = 0

        self.system_prefix_prompts = []
        self.num_system_prefix_prompts = 5
        self.concatenate_num = 1
        self.extra_needed_prompts = self.concatenate_num * self.num_system_prefix_prompts
        
        self.synthetic_multiturn_requests = None
        self.load_local = False

        with open('2108_dump_req_keys_0.pkl', 'rb') as f:
            self.stress_test_data = deque(pickle.load(f))
            print(f"len of data: {len(self.stress_test_data)}")
            for item in self.stress_test_data:
                print(f"item: {item}")

        if self.load_local and os.path.exists("synthetic_multiturn_256_requests.pkl"):
             with open('synthetic_multiturn_256_requests.pkl', 'rb') as f:
                self.synthetic_multiturn_requests = deque(pickle.load(f))
        else:
            if self.load_local and os.path.exists("candidate_inputs.pkl"):
                with open('candidate_inputs.pkl', 'rb') as f:
                    self.candidate_inputs = pickle.load(f)

            else:
                self.raw_candidate_inputs = sample_random_requests(
                    input_len=args.request_length,
                    output_len=args.output_length,
                    num_prompts=args.num_clients * args.num_rounds + self.extra_needed_prompts,
                    range_ratio=1.0,
                    tokenizer=self.tokenizer,
                    dataset_path=args.dataset_path,
                )
                for i in range(self.num_system_prefix_prompts):
                    self.system_prefix_prompts.append(self.raw_candidate_inputs[i * self.concatenate_num].prompt)
                    for j in range(1, self.concatenate_num):
                        self.system_prefix_prompts[-1] = self.system_prefix_prompts[-1] + self.raw_candidate_inputs[i * self.num_system_prefix_prompts + j].prompt

                self.candidate_inputs = []
                for i in range(self.concatenate_num * self.num_system_prefix_prompts, len(self.raw_candidate_inputs)):
                    random_idx = random.randint(0, len(self.system_prefix_prompts) - 1)
                    system_prefix_prompt = self.system_prefix_prompts[random_idx]
                    self.candidate_inputs.append(system_prefix_prompt + self.raw_candidate_inputs[i].prompt)
                    #print(f"i = {i}, init str = {str(self.candidate_inputs[-1])[:20]}")

                with open('candidate_inputs.pkl', 'wb') as f:
                    pickle.dump(self.candidate_inputs, f)

            init_requests = [
                (i, gen_payload(self.candidate_inputs[i], args.output_length))
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {"round": 0, "history": init_requests[i][1]["text"]}
                for i in range(args.num_clients)
            }
            self.ready_queue = []
            for i in range(self.num_clients):
                self.ready_queue.append(ReadyQueue(init_requests=init_requests[i: i+1]))
            self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds-10)
        self.performance_metrics = {"ttft": [], "latency": []}

    async def handle_request(self, item):
        try:
            request_history.append(item)
            client_id, payload = item
            #print(f"request len: {str(payload['text'])[:20]}")
            response = await async_request_sglang_generate(payload, self.url, self.pbar)
            if self.pbar.n >= self.pbar.total:
                self.finished_time = time.perf_counter()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed: {e}")

    def sync_request_sender(self):
        async def request_loop():
            while True:
                #print(f"sync send reqs")
                if len(self.synthetic_multiturn_requests) > 0 \
                    and self.sent_requests - self.completed_requests < args.max_parallel:
                    new_request = self.synthetic_multiturn_requests.popleft()
                    asyncio.create_task(self.handle_request(new_request))
                    self.sent_requests += 1
                    await asyncio.sleep(0.2)
                else:
                    await asyncio.sleep(0.05)
                    continue
                
                print(f"self.pbar.n : {self.pbar.n }")

                if self.pbar.n >= self.pbar.total:
                    break

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def request_sender(self):
        async def request_loop(idx):
            await asyncio.sleep(idx)  # Wait before sending the next request
            while True:
                current_client_id = None
                if self.sent_requests - self.completed_requests < args.max_parallel:
                    new_request = self.ready_queue[idx].pop()
                    if new_request:
                        current_client_id, _ = new_request
                        if current_client_id not in request_rate_map:
                            request_rate_map[current_client_id] = request_rate_list[idx % len(request_rate_list)]
                            client_id_to_idx[current_client_id] = idx
                            idx_to_client_id[idx] = current_client_id
                            #print(f"register, idx = {idx}, client_id = {current_client_id}")
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if self.pbar.n >= self.pbar.total:
                   break

                if current_client_id is None:
                    current_client_id = idx_to_client_id[idx]
                #print(f"client_id: {current_client_id}, request_rate: {request_rate_map[current_client_id]}, corres idx: {idx}")
                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = 10 #random.expovariate(min(request_rate_map[current_client_id], 30))
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / request_rate_map[current_client_id] if request_rate_map[current_client_id]> 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                #print(f"sleep_time: {sleep_time}")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

        async def main():
            tasks = [asyncio.create_task(request_loop(idx)) for idx in range(self.num_clients)]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
        loop.close()
        # Create and run the event loop for asynchronous requests
        #loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(loop)
        #loop.run_until_complete(request_loop())
        #loop.close()

    def sync_response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.completed_requests += 1
            except queue.Empty:
                if self.pbar.n >= self.pbar.total:
                    break

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.client_records[client_id]["history"] += response.generated_text
                self.client_records[client_id]["round"] += 1
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.completed_requests += 1

                if self.client_records[client_id]["round"] < args.num_rounds:
                    self.client_records[client_id][
                        "history"
                    ] += self.candidate_inputs.pop()
                    self.ready_queue[client_id_to_idx[client_id]].append(
                        (
                            client_id,
                            gen_payload(
                                self.client_records[client_id]["history"],
                                args.output_length,
                            ),
                        )
                    )
            except queue.Empty:
                if self.pbar.n >= self.pbar.total:
                    break

    def run(self):
        if self.synthetic_multiturn_requests is not None:
            print(f"sync send requests, total = {len(self.synthetic_multiturn_requests)}")
            sync_request_thread = threading.Thread(target=self.sync_request_sender, daemon=True)
            sync_response_thread = threading.Thread(target=self.sync_response_handler, daemon=True)

            self.start_time = time.perf_counter()
            sync_request_thread.start()
            sync_response_thread.start()

            sync_request_thread.join()
            sync_response_thread.join()
            self.pbar.close()
        else:
            request_thread = threading.Thread(target=self.request_sender, daemon=True)
            response_thread = threading.Thread(target=self.response_handler, daemon=True)

            self.start_time = time.perf_counter()
            request_thread.start()
            response_thread.start()

            request_thread.join()
            response_thread.join()
            self.pbar.close()

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": sorted(self.performance_metrics["ttft"])[
                    int(0.9 * len(self.performance_metrics["ttft"]))
                ],
                "median_ttft": sorted(self.performance_metrics["ttft"])[
                    len(self.performance_metrics["ttft"]) // 2
                ],
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": sorted(self.performance_metrics["latency"])[
                    int(0.9 * len(self.performance_metrics["latency"]))
                ],
                "median_latency": sorted(self.performance_metrics["latency"])[
                    len(self.performance_metrics["latency"]) // 2
                ],
                "throughput": self.pbar.total / (self.finished_time - self.start_time),
            },
        }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']}"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(
            f"  Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        log_to_jsonl_file(performance_data, args.log_file)

        if self.synthetic_multiturn_requests is None:
            num_req = len(request_history)
            with open(f"synthetic_multiturn_{num_req}_requests.pkl", 'wb') as f:
                pickle.dump(request_history, f)

if __name__ == "__main__":
    args = parse_args()
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    #for request_rate in [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
    #for request_rate in [16]:
    #    args.request_rate = request_rate
    requests.post(flush_cache_url)
    time.sleep(1)
    WorkloadGenerator(args).run()
