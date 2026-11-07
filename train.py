"""Reproducible 5→24→12→5 neural network and honest synthetic evaluation."""
from pathlib import Path
import json
import numpy as np
from vision import dataset, LABELS, FEATURES

ROOT = Path(__file__).parent

def train():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    x,y,groups = dataset()
    train_idx,test_idx = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(x,y,groups))
    scaler = StandardScaler().fit(x[train_idx])
    model = MLPClassifier(hidden_layer_sizes=(24,12),activation='relu',max_iter=250,
                          random_state=42,early_stopping=True,validation_fraction=.15)
    model.fit(scaler.transform(x[train_idx]), y[train_idx])
    predicted = model.predict(scaler.transform(x[test_idx]))
    weights = dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                   weights=[a.tolist() for a in model.coefs_],biases=[a.tolist() for a in model.intercepts_],labels=LABELS)
    report = dict(accuracy=accuracy_score(y[test_idx],predicted),f1_macro=f1_score(y[test_idx],predicted,average='macro'),
                  confusion_matrix=confusion_matrix(y[test_idx],predicted).tolist(),
                  classification_report=classification_report(y[test_idx],predicted,target_names=LABELS,output_dict=True),
                  loss=model.loss_curve_,train_samples=len(train_idx),test_samples=len(test_idx),seed=42,
                  features=FEATURES,architecture=[5,24,12,5],dataset='2500 imágenes sintéticas; 500 escenas; separación por escena',
                  limitation='Las métricas describen degradaciones sintéticas, no validación con fotografías reales.',
                  baseline_accuracy=float(np.mean(y[test_idx]==0)))
    (ROOT/'models/network.json').write_text(json.dumps(weights),encoding='utf-8')
    (ROOT/'reports/metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def predict(vector):
    data=json.loads((ROOT/'models/network.json').read_text(encoding='utf-8'))
    a=(np.asarray(vector)-data['mean'])/data['scale']
    for i,(w,b) in enumerate(zip(data['weights'],data['biases'])):
        a=a@np.asarray(w)+b
        if i<2: a=np.maximum(a,0)
    a=np.exp(a-a.max()); return a/a.sum()

if __name__=='__main__':
    r=train(); print(json.dumps({k:r[k] for k in ['accuracy','f1_macro','train_samples','test_samples']}))
