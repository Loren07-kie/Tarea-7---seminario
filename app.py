from pathlib import Path
from io import BytesIO
import base64,json,os
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask,request,jsonify,render_template
from vision import features,LABELS
from train import predict

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=8*1024*1024
Image.MAX_IMAGE_PIXELS=16_000_000
ROOT=Path(__file__).parent
TIPS=['Tu imagen tiene un balance adecuado. Prueba una edición creativa.',
      'Aumenta la iluminación antes de fotografiar. Prueba el filtro Luz.',
      'Reduce la exposición al capturar: los blancos recortados no se recuperan.',
      'Prueba Contraste para separar luces y sombras.',
      'Estabiliza la cámara y vuelve a enfocar. El detalle perdido no se puede recuperar.']

@app.get('/')
def index(): return render_template('index.html')

@app.get('/health')
def health(): return jsonify(status='ok',model=(ROOT/'models/network.json').exists())

@app.get('/api/metrics')
def metrics(): return jsonify(json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8')))

@app.post('/api/analyze')
def analyze():
    upload=request.files.get('image')
    mode=request.form.get('filter','original')
    if not upload: return jsonify(error='Selecciona una imagen.'),400
    if mode not in ['original','luz','contraste','grises','bordes','neon']:
        return jsonify(error='Filtro no válido.'),400
    try:
        with Image.open(upload.stream) as source:
            if source.width*source.height>16_000_000: raise ValueError()
            image=ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((1200,1200))
            rgb=np.asarray(image)
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
        return jsonify(error='Usa una imagen válida de hasta 16 megapíxeles y 8 MB.'),400
    vector=features(cv2.resize(rgb,(64,64),interpolation=cv2.INTER_AREA))
    probabilities=predict(vector); label=int(np.argmax(probabilities))
    result=rgb.copy()
    if mode=='luz': result=np.clip(rgb.astype(float)*1.35+12,0,255).astype('uint8')
    elif mode=='contraste': result=np.clip((rgb.astype(float)-128)*1.5+128,0,255).astype('uint8')
    elif mode=='grises': result=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    elif mode=='bordes': result=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),70,160)
    elif mode=='neon':
        edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),70,160)
        result=np.zeros_like(rgb); result[:,:,0]=edges; result[:,:,1]=edges//3; result[:,:,2]=edges
    buf=BytesIO(); Image.fromarray(result).save(buf,format='PNG')
    return jsonify(label=LABELS[label],tip=TIPS[label],probabilities=probabilities.tolist(),
                   features=vector.tolist(),image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())

@app.errorhandler(413)
def too_large(e): return jsonify(error='La imagen supera 8 MB.'),413

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Cache-Control']='no-store'
    response.headers['Referrer-Policy']='same-origin'
    return response

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
