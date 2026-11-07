"""Features shared by training and production; RGB input, no external API."""
import cv2
import numpy as np

LABELS = ['Equilibrada', 'Oscura', 'Sobreexpuesta', 'Bajo contraste', 'Desenfocada']
FEATURES = ['brillo', 'contraste', 'nitidez', 'sombras', 'altas luces']

def features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.array([gray.mean()/255, gray.std()/128,
                     np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())/12,
                     (gray < 35).mean(), (gray > 225).mean()])

def dataset(seed=42, groups=500):
    """Five controlled degradations per procedural scene. Split by scene!"""
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for group in range(groups):
        base = rng.integers(45, 211, (64,64,3), dtype=np.uint8)
        for _ in range(8):
            x,y = rng.integers(0,48,2)
            cv2.rectangle(base,(int(x),int(y)),(int(x+12),int(y+12)),tuple(map(int,rng.integers(30,225,3))),-1)
        variants = [base, (base*rng.uniform(.15,.35)).astype('uint8'),
                    np.clip(base*rng.uniform(.25,.5)+rng.uniform(180,205),0,255).astype('uint8'),
                    np.clip(128+(base.astype(float)-128)*rng.uniform(.08,.2),0,255).astype('uint8'),
                    cv2.GaussianBlur(base,(15,15),rng.uniform(3,6))]
        for label,img in enumerate(variants):
            xs.append(features(img)); ys.append(label); ids.append(group)
    return np.array(xs), np.array(ys), np.array(ids)
