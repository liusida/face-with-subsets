import torchvision
# from models.pl_image_classifier import LitImageClassifier
# from models.pl_transfer_densenet import LitImageClassifier
from models.pl_transfer_vggface2 import LitImageClassifier

import pickle, argparse
import cv2
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", default=None, help="file path")
args = parser.parse_args()

with open("targets_rev.pkl", "rb") as f:
    targets_rev = pickle.load(f) # {'0-2':0, ...}

if args.filename is None:
    df_labels = pd.read_csv("./data-labels/ffhq_aging_labels.csv")
    truth = []
    filenames = []
    for i in range(100):
        filename = f"data/small/{i:05}.png"
        filenames.append(filename)
        t = df_labels.iloc[i]['age_group']
        truth.append(t)
else:
    filenames = [args.filename]
    truth = [""]

def cv2_put_text(image, txt, org, color):
    image = cv2.putText(image, txt, org=org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.4, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA) #color in BGR
    image = cv2.putText(image, txt, org=org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.4, color=color, thickness=1, lineType=cv2.LINE_AA) #color in BGR
    return image

for id, filename in enumerate(filenames):
    output_filename = filename.replace('/', '_')

    x = torchvision.datasets.folder.pil_loader(filename)
    x = torchvision.transforms.ToTensor()(x)
    x = torchvision.transforms.Resize([256,256])(x)
    x = x.unsqueeze(0)
    print(x.shape)

    model = LitImageClassifier.load_from_checkpoint("checkpoints/vggface2.ckpt").eval()
    # model = LitImageClassifier.load_from_checkpoint("checkpoints/tmp.ckpt")

    y_hat = model(x)
    y_hat = y_hat.flatten().detach().numpy().tolist()
    report = ""
    min_prob = 0
    for i in range(len(y_hat)):
        if y_hat[i]>min_prob:
            min_prob = y_hat[i]
            report = f"{targets_rev[i]} : {y_hat[i]:.0%}"
    print(report)
    image = cv2.imread(filename)
    image = cv2_put_text(image, report, org=(0, 15), color=(255,0,0)) #color in BGR

    txt_truth = f"truth: {truth[id]}"
    image = cv2_put_text(image, txt_truth, org=(0, 30), color=(0,0,0)) #color in BGR
    
    cv2.imwrite(filename=f"output/{output_filename}", img=image)
