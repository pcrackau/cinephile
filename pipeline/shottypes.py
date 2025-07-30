
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.datasets import ImageFolder

NUM_CLASSES = 5

def predict_shottype(sample_image):

    # TODO hardcoded. Change later
    class_names = ['close_up', 'extreme_close_up', 'long_shot', 'medium_long_shot', 'medium_shot']

    # Load model
    model = models.vgg16_bn(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, len(class_names))
    model.load_state_dict(torch.load("models/shottypes_vgg16_bn.pt", map_location="cpu"))
    model.eval()

    # Preprocess image, match finetuning
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load images
    img_path = sample_image
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = probs.argmax().item()

    print(f"Shottype: {class_names[pred_class]}")
    return f"{class_names[pred_class]}"