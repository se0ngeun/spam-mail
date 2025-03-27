from flask import Flask, render_template, request, jsonify  # render_template 추가
from collections import OrderedDict
import sqlite3
import joblib

app = Flask(__name__)

# 모델 로딩
model = joblib.load('spam_classifier_rf.pkl')
label_mapping = {1: '업무', 2: '광고', 3: '기타'}

# SQLite 데이터베이스 초기화 (category 필드 추가)
def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    # 테이블이 존재하면 삭제
    cursor.execute("DROP TABLE IF EXISTS feedback")
    
    # 새 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_content TEXT,
            prediction TEXT,
            category TEXT,  -- 카테고리 필드 추가
            user_feedback TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('category.html')  # index 페이지에서 category.html을 렌더링

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.get_json()
    email_content = data.get('email_text', '')

    # 예측
    probabilities = model.predict_proba([email_content])[0]
    prediction = model.predict([email_content])[0]
    
    spam_prob = probabilities[1]  # 스팸 확률
    normal_prob = probabilities[0]  # 정상 확률
    result = "스팸" if prediction == 1 else "정상"
    predicted_category = label_mapping.get(prediction, '기타')  # 예측된 카테고리

    return jsonify({
        'result': result,
        'type': predicted_category,
        'spam_prob': spam_prob,
        'normal_prob': normal_prob
    })

@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    email_content = data.get('email_content', '')
    prediction = data.get('prediction', '')
    category = data.get('category', '')  # 클라이언트로부터 받은 카테고리
    user_feedback = data.get('user_feedback', '')

    # DB에 저장
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (email_content, prediction, category, user_feedback)
        VALUES (?, ?, ?, ?)
    ''', (email_content, prediction, category, user_feedback))
    conn.commit()
    conn.close()

    return jsonify({'message': '피드백이 저장되었습니다.'})

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()

    # 카테고리, 예측 결과, 사용자 피드백 순으로 정렬
    cursor.execute("SELECT email_content, prediction, category, user_feedback FROM feedback ORDER BY category, prediction, user_feedback")
    rows = cursor.fetchall()
    conn.close()

    feedback_list = []
    for row in rows:
        feedback = OrderedDict([
            ("email_content", row[0]),   # 이메일 내용
            ("prediction", row[1]),      # 예측 결과
            ("category", row[2]),        # 카테고리
            ("user_feedback", row[3])    # 피드백
        ])
        feedback_list.append(feedback)
    
    return jsonify(feedback_list)


if __name__ == '__main__':
    app.run(debug=True)
