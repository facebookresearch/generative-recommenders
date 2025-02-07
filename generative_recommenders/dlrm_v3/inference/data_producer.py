# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import logging
import threading
import time
from queue import Queue
from typing import List, Optional

import torch
from generative_recommenders.dlrm_v3.datasets.dataset import Dataset, Samples

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("data_producer")


class QueryItem:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_ids: List[int], samples: Samples) -> None:
        self.query_ids = query_ids
        self.samples = samples
        self.start: float = time.time()


class SingleThreadDataProducer:
    def __init__(self, ds: Dataset, run_one_item) -> None:  # pyre-ignore [2]
        self.ds = ds
        self.run_one_item = run_one_item  # pyre-ignore [4]

    def enqueue(self, query_ids: List[int], content_ids: List[int]) -> None:
        with torch.profiler.record_function("data batching"):
            samples = self.ds.get_samples(content_ids)
            self.run_one_item(QueryItem(query_ids, samples))

    def finish(self) -> None:
        pass


class MultiThreadDataProducer:
    def __init__(
        self,
        ds: Dataset,
        threads: int,
        run_one_item,  # pyre-ignore [2]
    ) -> None:
        queue_size_multiplier = 4
        self.ds = ds
        self.threads = threads
        self.run_one_item = run_one_item  # pyre-ignore [4]
        self.tasks: Queue[Optional[QueryItem]] = Queue(
            maxsize=threads * queue_size_multiplier
        )
        self.workers: List[threading.Thread] = []
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue: Queue[Optional[QueryItem]]) -> None:
        stream = torch.cuda.Stream()
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                tasks_queue.task_done()
                break
            with torch.inference_mode(), torch.cuda.stream(stream):
                self.run_one_item(qitem)
            tasks_queue.task_done()

    def enqueue(self, query_ids: List[int], content_ids: List[int]) -> None:
        with torch.profiler.record_function("data batching"):
            samples = self.ds.get_samples(content_ids)
            self.tasks.put(QueryItem(query_ids, samples))

    def finish(self) -> None:
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()
