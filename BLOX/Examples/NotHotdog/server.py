from flask import Flask, render_template, request,Response
import requests
from werkzeug import secure_filename
app = Flask(__name__,template_folder='templates/')
DIST = ["Hotdog", "Not hotdog"]
from torchvision import transforms
import io,base64
from PIL import Image
import torch,json

transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
def img2tensor(image, transform_img=True):
    if isinstance(image,torch.Tensor):return image
    image = Image.open(image if isinstance(image,str) else io.BytesIO(image) )
    image = image.resize([224, 224], Image.LANCZOS)

    if transform_img:
        image = transform(image).unsqueeze(0)
        image = image[:,:3,:,:]
    return image
@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    r = ''
    f = None
    if request.method == 'POST':
        try:
            f = request.files['file']
            i = img2tensor(f.read() ).data.numpy().tolist() 
            _r = requests.post('http://127.0.0.1:8000/predict',{'data': i })
            _r = json.loads(_r.content.decode())
            a = json.loads(_r['resp'])
            r = DIST[0] if a[0] > a[1] else DIST[1]
        except Exception as e:
            r = "Trouble processing image"
    return Response(render_template('index.html', t=r,image=base64.b64encode(f.read() ) if f else ''))
		
if __name__ == '__main__':
   app.run(debug=True)