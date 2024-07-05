
from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Item
from .serializers import ItemSerializer
import requests
import torch
import torchvision
from torchvision import transforms
from typing import List, Tuple
from PIL import Image
from torch import nn
import urllib.request
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

@api_view(['GET'])
def get_data(req):
    items = Item.objects.all()
    serializer = ItemSerializer(items, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def add_data(req):
    serializer = ItemSerializer(data=req.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)


@api_view(['POST'])
def classification(req):
    ctscanUrl = req.data['ctscanUrl']

    result, accuracy = checkCtscan(ctscanUrl)
    
    if result != "Nonctscan":
        result, accuracy = predict(ctscanUrl)

    image_name = ctscanUrl.split('/')
    if os.path.exists(f"./static/ctImages/{image_name[4]}"):
        os.remove(f"./static/ctImages/{image_name[4]}")
    else:
        print("The file does not exist")

    resObj = {"ctscanUrl": ctscanUrl, "result": result, "accuracy": accuracy}

    return Response(resObj)

def predict(ctscanUrl):
    custom_image_path = ctscanUrl
    model_name = "modelv2.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = torchvision.models.vit_b_16().to(device)
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventicular', 'Normal', 'Subarachnoid', 'Subdural']
    model_file.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    model_file.load_state_dict(torch.load(f"./api/models/{model_name}", torch.device('cpu')))
        
    result, accuracy = pred_and_plot_image(model=model_file,
                        image_path=custom_image_path,
                        class_names=class_names)
        
    accuracy = float(accuracy) * 100
    
    return result, accuracy

def checkCtscan(ctscanUrl):
    custom_image_path = ctscanUrl
    model_name = "model-ctscan.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = torchvision.models.vit_b_16().to(device)
    class_names = ['Ctscan', 'Nonctscan']
    model_file.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    model_file.load_state_dict(torch.load(f"./api/models/{model_name}", torch.device('cpu')))

    result, accuracy = pred_and_plot_image(model=model_file,
                        image_path=custom_image_path,
                        class_names=class_names)
        
    accuracy = float(accuracy) * 100
    
    return result, accuracy

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    # if image_path from a url
    image_name = image_path.split('/')
    # urllib.request.urlretrieve(image_path, "ctImage.png") 
    urllib.request.urlretrieve(image_path, f"./static/ctImages/{image_name[4]}") 
    # img = Image.open("./static/ctImages/ctImage.png")
    img = Image.open(f"./static/ctImages/{image_name[4]}")

    img = img.convert("RGB")

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),

            ]
        )

    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_acc = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_acc, dim=1)
    
    return class_names[target_image_pred_label], f"{target_image_pred_acc.max():.3f}"