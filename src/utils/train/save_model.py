from queue import Queue
from threading import Thread

from rich import print
from safetensors.torch import save_file


class ModelSaver:
    def __init__(self) -> None:
        self.queue = Queue(maxsize=5)
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def add(self, model_dict: dict, path: str) -> None:
        if self.queue.full():
            print("[red]ModelSaver queue is full, waiting to save...[/red]")
        self.queue.put((model_dict, path))  # fullならブロックする

    def _save_model(self, model_dict: dict, path: str):
        save_file(model_dict, path)
        print(f"[green]Model saved to {path}[/green]")

    def _worker(self) -> None:
        while True:
            print(f"rate of saving: {self.queue.qsize()} models in queue")
            model, path = self.queue.get(block=True)  # block=Trueで待機
            try:
                self._save_model(model, path)
            except Exception as e:
                print(f"Error saving model to {path}: {e}")
            finally:
                self.queue.task_done()  # 保存完了を通知
