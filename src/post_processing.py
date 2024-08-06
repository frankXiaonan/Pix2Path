from tensorboard import program
from utils import LOGS_FIT_DIR

def post_process():
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", LOGS_FIT_DIR])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")