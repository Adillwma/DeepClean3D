import os
import time
import subprocess
import webbrowser
from torch.utils.tensorboard import SummaryWriter


def launch_tensorboard(logdir="runs", port=6006, open_browser=True, wait_time=3):
    """
    Launch TensorBoard in the background and open it in the default browser.
    """
    os.makedirs(logdir, exist_ok=True)

    # Kill old TensorBoard (avoid "port in use" errors)
    try:
        if os.name == "nt":  # Windows
            subprocess.run("taskkill /F /IM tensorboard.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:  # macOS/Linux
            subprocess.run("pkill -f tensorboard", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Launch TensorBoard
    process = subprocess.Popen([
        "tensorboard",
        f"--logdir={logdir}",
        "--host=localhost",
        f"--port={port}"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # if open_browser:
    #     time.sleep(wait_time)
    #     webbrowser.open_new_tab(f"http://localhost:{port}")

    # print(f"✅ TensorBoard launched at http://localhost:{port} (logdir: {logdir})")
    return process

class TensorBoardLogger:
    """
    Wrapper class for TensorBoard logging in PyTorch training.
    """

    def __init__(self, logdir="runs/experiment", rootdir="runs/", port=6006, auto_launch=True):
        self.logdir = logdir
        self.root_dir = rootdir
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.tb_process = None
        self.port = port

        if auto_launch:
            self.tb_process = launch_tensorboard(logdir=self.root_dir, port=port)

    def launch_dashboard(self):
        webbrowser.open_new_tab(f"http://localhost:{self.port}")
        print(f"Results Dashboard launched at http://localhost:{self.port}")

    def log_performance_metrics(self, epoch, loss=None, accuracy=None, lr=None, **kwargs):
        """
        Log common metrics (loss, accuracy, learning rate, etc.).
        Additional named metrics can be passed as keyword arguments.
        """
        if loss is not None:
            self.writer.add_scalar("Loss/train", loss, epoch)
        if accuracy is not None:
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)
        if lr is not None:
            self.writer.add_scalar("LearningRate", lr, epoch)
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, epoch)

    def add_text(self, tag, text, global_step=0):
        self.writer.add_text(tag, text, global_step)

    def add_graph(self, model, example_input):
        """
        Optionally log model architecture.
        """
        try:
            self.writer.add_graph(model, example_input)
        except Exception as e:
            print(f"⚠️ Could not log model graph: {e}")

    def add_histogram(self, tag, values, global_step=0, bins=100):
        self.writer.add_histogram(tag, values, global_step, bins=bins)

    def add_image(self, tag, img_tensor, global_step=0):
        self.writer.add_image(tag, img_tensor, global_step)

    def add_images(self, tag, img_tensor, global_step=0):
        self.writer.add_images(tag, img_tensor, global_step)

    def add_figure(self, tag, figure, global_step=0):
        self.writer.add_figure(tag, figure, global_step)

    def add_scalar(self, tag, scalar_value, global_step=0):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_hparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """
        Flush and close the writer, terminate TensorBoard.
        """
        self.writer.flush()
        self.writer.close()
        if self.tb_process:
            self.tb_process.terminate()
            print("TensorBoard process terminated.")

# # Example usage
# if __name__ == "__main__":
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim

#     # Dummy data and model
#     x = torch.randn(100, 10)
#     y = torch.randint(0, 2, (100,))
#     model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)

#     # Create logger (auto-launches TensorBoard)
#     logger = TensorBoardLogger(logdir="tensortestboard/exp1", port=6006)

#     # Optionally log text or model graph
#     logger.add_text("Experiment Name", "Test Experiment with Live TensorBoard", 0)
#     logger.add_model_graph(model, x[:1])

#     # Training loop
#     for epoch in range(1, 60000):
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         acc = (out.argmax(1) == y).float().mean()
#         loss.backward()
#         optimizer.step()

#         # Log metrics
#         logger.log_metrics(epoch, loss=loss.item(), accuracy=acc.item(), lr=optimizer.param_groups[0]["lr"])
#         print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")

#     logger.close()

