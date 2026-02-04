from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

def log_val_predictions_to_tensorboard(cfg, model, data_loader, tb_dir, max_images=5):
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    for idx, inputs in enumerate(data_loader):
        if idx >= max_images:
            break
        outputs = model(inputs)
        v = Visualizer(inputs[0]["image"].permute(1, 2, 0).cpu().numpy(), metadata=metadata)
        out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        img = out.get_image()
        writer.add_image(f"val_prediction_{idx}", img, 0, dataformats="HWC")

    writer.close()


def log_image_to_tensorboard(image_path, tag, log_dir, step=0):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    writer = SummaryWriter(log_dir)
    writer.add_image(tag, img_rgb.transpose(2, 0, 1), global_step=step)
    writer.close()

def logText(cfg, log_dir, subject="TrainingParameters"):
    try:
        log_path = os.path.join(log_dir, "tensorboard")
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
        params_str = cfg.dump()  # ✅ cleaner than manual looping
        writer.add_text(subject, f"```\n{params_str}\n```", global_step=0)
        writer.flush()
        writer.close()
    except Exception as e:
        print(f"[logText] Failed to write text summary: {e}")
