# คู่มือการทดสอบ AI ตัวใหม่กับ Dataset Phishing Email

## ภาพรวม
สคริปต์ `test_new_ai.py` ใช้สำหรับทดสอบ AI ตัวใหม่ของคุณกับ dataset Phishing Email ที่มี 18,650 ตัวอย่าง

## ข้อมูล Dataset
- **จำนวนตัวอย่าง**: 18,650 ตัวอย่าง
- **Phishing Email**: 7,328 ตัวอย่าง (39.3%)
- **Safe Email**: 11,322 ตัวอย่าง (60.7%)

## วิธีการใช้งาน

### 1. เปิดไฟล์ `test_new_ai.py`

### 2. แก้ไขฟังก์ชัน `your_new_ai_predict(texts)`

```python
def your_new_ai_predict(texts):
    """
    ฟังก์ชันสำหรับทดสอบ AI ตัวใหม่ของคุณ
    """
    predictions = []
    confidences = []
    
    for text in texts:
        # ใส่โค้ด AI ตัวใหม่ของคุณที่นี่
        # ตัวอย่าง:
        
        # วิธีที่ 1: ใช้โมเดลที่บันทึกไว้
        # model = joblib.load('your_model.pkl')
        # pred = model.predict([text])[0]
        # conf = model.predict_proba([text]).max()
        
        # วิธีที่ 2: ใช้ API
        # response = your_api.predict(text)
        # pred = response['prediction']
        # conf = response['confidence']
        
        # วิธีที่ 3: ใช้โมเดล Deep Learning
        # pred = your_model.predict(text)
        # conf = your_model.predict_proba(text).max()
        
        if pred == 1:  # หรือเงื่อนไขอื่นๆ
            predictions.append('Phishing Email')
            confidences.append(conf)
        else:
            predictions.append('Safe Email')
            confidences.append(conf)
    
    return predictions, confidences
```

### 3. รันสคริปต์
```bash
python test_new_ai.py
```

## ผลลัพธ์ที่ได้

### 1. ข้อมูลแสดงผล
- **ความแม่นยำโดยรวม** (Overall Accuracy)
- **ประสิทธิภาพแยกตามคลาส**:
  - Precision (ความแม่นยำ)
  - Recall (ความครอบคลุม)
  - F1-Score (คะแนนรวม)
  - Support (จำนวนตัวอย่าง)
  - ความมั่นใจเฉลี่ย

### 2. ไฟล์ผลลัพธ์
- `ai_predictions.csv` - การทำนายของ AI
- `ai_evaluation_report.json` - รายงานการประเมิน

## ตัวอย่างการใช้งาน AI ต่างๆ

### 1. Machine Learning Models
```python
from joblib import load

def your_new_ai_predict(texts):
    # โหลดโมเดล
    model = load('your_model.joblib')
    vectorizer = load('your_vectorizer.joblib')
    
    predictions = []
    confidences = []
    
    for text in texts:
        # แปลงข้อความ
        X = vectorizer.transform([text])
        
        # ทำนาย
        pred = model.predict(X)[0]
        conf = model.predict_proba(X).max()
        
        if pred == 1:
            predictions.append('Phishing Email')
        else:
            predictions.append('Safe Email')
        
        confidences.append(conf)
    
    return predictions, confidences
```

### 2. Deep Learning Models
```python
import tensorflow as tf

def your_new_ai_predict(texts):
    # โหลดโมเดล
    model = tf.keras.models.load_model('your_model.h5')
    
    predictions = []
    confidences = []
    
    for text in texts:
        # Preprocess ข้อความ
        processed_text = preprocess_text(text)
        
        # ทำนาย
        pred_proba = model.predict([processed_text])[0]
        pred = 1 if pred_proba > 0.5 else 0
        conf = max(pred_proba)
        
        if pred == 1:
            predictions.append('Phishing Email')
        else:
            predictions.append('Safe Email')
        
        confidences.append(conf)
    
    return predictions, confidences
```

### 3. Pre-trained Language Models
```python
from transformers import pipeline

def your_new_ai_predict(texts):
    # โหลดโมเดล
    classifier = pipeline("text-classification", model="your_model")
    
    predictions = []
    confidences = []
    
    for text in texts:
        # ทำนาย
        result = classifier(text)
        
        if result['label'] == 'LABEL_1':
            predictions.append('Phishing Email')
        else:
            predictions.append('Safe Email')
        
        confidences.append(result['score'])
    
    return predictions, confidences
```

### 4. API-based Models
```python
import requests

def your_new_ai_predict(texts):
    predictions = []
    confidences = []
    
    for text in texts:
        # เรียก API
        response = requests.post('your_api_endpoint', json={'text': text})
        result = response.json()
        
        if result['prediction'] == 'phishing':
            predictions.append('Phishing Email')
        else:
            predictions.append('Safe Email')
        
        confidences.append(result['confidence'])
    
    return predictions, confidences
```

## เกณฑ์การประเมิน

### เป้าหมายที่แนะนำ
- **Accuracy > 90%** - ความแม่นยำโดยรวม
- **Phishing Email Recall > 95%** - ตรวจจับ Phishing Email ได้ดี
- **Phishing Email Precision > 85%** - จำแนก Phishing Email ได้แม่นยำ
- **F1-Score > 90%** - คะแนนรวม

### การปรับปรุง
หากผลลัพธ์ไม่เป็นไปตามเป้าหมาย:
1. ตรวจสอบคุณภาพของข้อมูลฝึก
2. ปรับปรุง feature engineering
3. ลองใช้โมเดลที่แตกต่างกัน
4. ปรับ hyperparameters

## ข้อควรระวัง

1. **ความมั่นใจ (Confidence)**: ควรอยู่ในช่วง 0-1
2. **Label Format**: ต้องใช้ 'Phishing Email' และ 'Safe Email' เท่านั้น
3. **Memory Usage**: ข้อมูลมีขนาดใหญ่ ควรจัดการ memory ให้ดี
4. **Error Handling**: ควรจัดการ error ที่อาจเกิดขึ้น

## การแก้ไขปัญหา

### ปัญหาที่พบบ่อย
1. **Import Error**: ตรวจสอบว่าได้ติดตั้ง library ที่จำเป็นแล้ว
2. **Memory Error**: ลดขนาดข้อมูลหรือใช้ batch processing
3. **Model Loading Error**: ตรวจสอบ path ของไฟล์โมเดล
4. **Prediction Error**: ตรวจสอบ format ของ input และ output

---
*คู่มือนี้จะช่วยให้คุณทดสอบ AI ตัวใหม่ได้อย่างมีประสิทธิภาพ*
