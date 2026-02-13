import torch.cuda
from fastapi import FastAPI , File , UploadFile , HTTPException
import uvicorn
import torch.nn as nn
from torchvision import transforms
import io
from PIL import Image
import streamlit as st

transform_data = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

class ModelM(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*64*64 ,256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )
  def forward(self, image):
    image = self.first(image)
    image =self.second(image)
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelM()
model.load_state_dict(torch.load('model_fl.pth' , map_location=device))
model.to(device)
model.eval()
flowers_app = FastAPI()
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@flowers_app.post('/predict')
async def check_flowers(file : UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=404 , detail='File not found')
        img_open = Image.open(io.BytesIO(data))
        img_ten = transform_data(img_open).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_ten)
            pred =y_pred.argmax(dim =1).item()
            return {'answer' : classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))



if __name__ == '__main__':
    uvicorn.run(flowers_app , port = 8000 , host = '127.0.0.1')