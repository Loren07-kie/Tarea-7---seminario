import unittest
from io import BytesIO
import numpy as np
from PIL import Image
from app import app
from vision import dataset,features
from train import predict

class LaboratoryTests(unittest.TestCase):
    def setUp(self): self.client=app.test_client()
    def photo(self):
        b=BytesIO();Image.fromarray(np.random.default_rng(1).integers(0,256,(64,64,3),dtype='uint8')).save(b,format='PNG');b.seek(0);return b
    def test_health_and_page(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertTrue(self.client.get('/health').json['model'])
    def test_all_filters_return_decodable_png_and_probabilities(self):
        import base64
        for mode in ['original','luz','contraste','grises','bordes','neon']:
            r=self.client.post('/api/analyze',data={'image':(self.photo(),'photo.png'),'filter':mode})
            self.assertEqual(r.status_code,200)
            self.assertAlmostEqual(sum(r.json['probabilities']),1)
            Image.open(BytesIO(base64.b64decode(r.json['image'].split(',')[1]))).verify()
    def test_invalid_requests(self):
        self.assertEqual(self.client.post('/api/analyze').status_code,400)
        self.assertEqual(self.client.post('/api/analyze',data={'image':(BytesIO(b'fake'),'fake.png')}).status_code,400)
        self.assertEqual(self.client.post('/api/analyze',data={'image':(self.photo(),'x.png'),'filter':'evil'}).status_code,400)
    def test_oversize(self):
        self.assertEqual(self.client.post('/api/analyze',data={'image':(BytesIO(b'x'*(8*1024*1024+1)),'big.png')}).status_code,413)
    def test_unseen_synthetic_scenes(self):
        x,y,_=dataset(seed=2026,groups=30)
        predictions=np.array([predict(v).argmax() for v in x])
        self.assertGreater(float((predictions==y).mean()),.9)
    def test_blank_images_numerical_stability(self):
        for value in [0,128,255]:
            p=predict(features(np.full((64,64,3),value,dtype='uint8')))
            self.assertTrue(np.isfinite(p).all()); self.assertAlmostEqual(p.sum(),1)

if __name__=='__main__': unittest.main()
