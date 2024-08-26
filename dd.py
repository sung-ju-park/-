import os                                            # 운영 체제와 상호 작용하기 위한 모듈
import numpy as np                                   # 수치 연산을 위한 파이썬 라이브러리
import sqlite3                                       # SQLite 데이터베이스와의 상호작용을 위한 모듈
from mtcnn import MTCNN                              # 얼굴 탐지를 위한 MTCNN 모델
from PIL import Image                                # 이미지 처리 및 조작을 위한 PIL 라이브러리
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # VGG16 모델과 입력 전처리 함수
from tensorflow.keras.preprocessing import image     # 이미지를 로드하고 전처리하는 데 사용되는 모듈
from scipy.spatial.distance import cosine            # 코사인 유사도(두 벡터 사이의 유사도를 측정하는 방법 중 하나)를 계산하기 위한 함수

# 1. 데이터 전처리
def preprocess_images(input_dir, output_dir):
    detector = MTCNN()                               # MTCNN 얼굴 탐지기 초기화
    for celeb_name in os.listdir(input_dir):         # 입력 디렉토리의 모든 연예인 폴더 반복
        celeb_dir = os.path.join(input_dir, celeb_name)  # 각 연예인 폴더의 경로 생성
        if not os.path.isdir(celeb_dir):             # 폴더가 아닌 경우 건너뜀
            continue
        output_celeb_dir = os.path.join(output_dir, celeb_name)  # 출력 디렉토리에 동일한 연예인 폴더 생성
        os.makedirs(output_celeb_dir, exist_ok=True) # 출력 폴더 생성 (이미 존재하면 무시)

        for img_name in os.listdir(celeb_dir):       # 각 연예인 폴더의 모든 이미지 파일 반복
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일인 경우에만 처리
                img_path = os.path.join(celeb_dir, img_name)  # 이미지 파일의 전체 경로 생성
                try:
                    img = Image.open(img_path).convert('RGB')  # 이미지를 열어 RGB 형식으로 변환
                    img_array = np.array(img)                 # 이미지를 NumPy 배열로 변환
                    faces = detector.detect_faces(img_array)  # 얼굴 탐지
                    if faces:                                 # 얼굴이 탐지된 경우
                        x, y, w, h = faces[0]['box']          # 첫 번째 얼굴의 위치 및 크기 추출
                        face = img.crop((x, y, x+w, y+h))     # 얼굴 영역을 잘라냄
                        resized_face = face.resize((224, 224))  # 얼굴 이미지를 224x224 크기로 리사이즈
                        output_path = os.path.join(output_celeb_dir, img_name)  # 출력 경로 생성
                        resized_face.save(output_path)        # 리사이즈된 얼굴 이미지를 저장
                        print(f"Successfully processed: {img_path}")  # 성공 메시지 출력
                    else:
                        print(f"No face detected in {img_path}")  # 얼굴이 감지되지 않음
                except Exception as e:                        # 오류 발생 시 예외 처리
                    print(f"Error processing {img_path}: {str(e)}")  # 오류 메시지 출력

# 얼굴 특징 추출
def extract_features(img_path):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # VGG16 모델 로드, 최상위 층 제거, 풀링 적용
    img = image.load_img(img_path, target_size=(224, 224))  # 이미지를 224x224 크기로 로드
    img_array = image.img_to_array(img)                    # 이미지를 배열로 변환
    expanded_img = np.expand_dims(img_array, axis=0)       # 차원을 추가하여 배치 형태로 변환
    preprocessed_img = preprocess_input(expanded_img)      # VGG16 모델에 맞게 전처리
    features = model.predict(preprocessed_img)             # 모델을 통해 특징 벡터 추출
    return features.flatten()                              # 1차원 배열로 반환

# 데이터베이스 구축
def create_database():
    db_path = os.path.join(os.path.expanduser('~'), 'celebrity_features.db')  # 사용자 홈 디렉토리에 데이터베이스 파일 경로 생성
    conn = sqlite3.connect(db_path)                    # SQLite 데이터베이스에 연결
    c = conn.cursor()                                  # 커서 객체 생성
    c.execute('''CREATE TABLE IF NOT EXISTS features
                 (id INTEGER PRIMARY KEY, celeb_name TEXT, feature_vector BLOB)''')  # 'features' 테이블 생성
    conn.commit()                                      # 변경 사항 저장
    conn.close()                                       # 데이터베이스 연결 종료
    return db_path                                     # 데이터베이스 경로 반환

def insert_features(db_path, celeb_name, feature_vector):
    conn = sqlite3.connect(db_path)                    # SQLite 데이터베이스에 연결
    c = conn.cursor()                                  # 커서 객체 생성
    c.execute("INSERT INTO features (celeb_name, feature_vector) VALUES (?, ?)",
              (celeb_name, feature_vector.tobytes()))  # 연예인 이름과 특징 벡터를 테이블에 삽입
    conn.commit()                                      # 변경 사항 저장
    conn.close()                                       # 데이터베이스 연결 종료

# 유사도 비교 알고리즘
def compute_similarity(feature1, feature2):
    return 1 - cosine(feature1, feature2)              # 코사인 유사도를 통해 두 벡터의 유사도 계산

# 연예인 닮은꼴 찾기
def find_lookalike(db_path, input_features):
    conn = sqlite3.connect(db_path)                    # SQLite 데이터베이스에 연결
    c = conn.cursor()                                  # 커서 객체 생성
    c.execute("SELECT celeb_name, feature_vector FROM features")  # 모든 연예인의 이름과 특징 벡터를 가져옴
    results = c.fetchall()                             # 결과를 가져옴
    conn.close()                                       # 데이터베이스 연결 종료

    max_similarity = 0                                 # 초기 최대 유사도 설정
    lookalike = None                                   # 초기 닮은꼴 연예인 설정
    for celeb_name, feature_bytes in results:          # 각 연예인의 특징 벡터와 비교
        db_features = np.frombuffer(feature_bytes, dtype=np.float32)  # 데이터베이스의 특징 벡터를 NumPy 배열로 변환
        similarity = compute_similarity(input_features, db_features)  # 입력된 이미지와 데이터베이스 이미지의 유사도 계산
        if similarity > max_similarity:                # 더 높은 유사도가 발견된 경우
            max_similarity = similarity                # 최대 유사도 갱신
            lookalike = celeb_name                     # 닮은꼴 연예인 갱신

    return lookalike, max_similarity * 100             # 닮은꼴 연예인과 유사도 반환

# 메인 실행 함수
def main():
    # 1. 데이터베이스 생성
    print("데이터베이스 생성 중...")                  # 데이터베이스 생성 메시지 출력
    db_path = create_database()                        # 데이터베이스 생성

    # 2. 특징 추출 및 데이터베이스 구축
    print("특징 추출 및 데이터베이스 구축 중...")      # 특징 추출 및 데이터베이스 구축 메시지 출력
    processed_dir = 'processed_images'                 # 전처리된 이미지가 있는 디렉토리 경로
    
    # 총 이미지 수 계산
    total_images = sum([len(files) for r, d, files in os.walk(processed_dir)])  # 디렉토리 내 모든 이미지 수 계산
    processed_images = 0                               # 처리된 이미지 수 초기화

    for celeb_name in os.listdir(processed_dir):       # 각 연예인 폴더 반복
        celeb_dir = os.path.join(processed_dir, celeb_name)  # 연예인 폴더 경로 생성
        if os.path.isdir(celeb_dir):                   # 폴더인지 확인
            for img_name in tqdm(os.listdir(celeb_dir), desc=f"처리 중: {celeb_name}", unit="image"):  # 폴더 내 이미지 파일 반복
                img_path = os.path.join(celeb_dir, img_name)  # 이미지 파일 경로 생성
                features = extract_features(img_path)         # 이미지에서 특징 벡터 추출
                insert_features(db_path, celeb_name, features)  # 특징 벡터를 데이터베이스에 삽입
                processed_images += 1                          # 처리된 이미지 수 증가
                print(f"\r전체 진행률: {processed_images}/{total_images} ({processed_images/total_images*100:.2f}%)", end="")  # 진행률 출력
    
    print("\n데이터베이스 구축 완료")                   # 데이터베이스 구축 완료 메시지 출력

    # 3. 테스트
    while True:
        print("\n테스트 시작...")                      # 테스트 시작 메시지 출력
        test_image_path = input("테스트할 이미지 경로를 입력하세요 (종료하려면 'q' 입력): ")  # 테스트할 이미지 경로 입력 요청
        if test_image_path.lower() == 'q':             # 'q' 입력 시 종료
            break
        if os.path.exists(test_image_path):            # 이미지 경로가 유효한지 확인
            test_features = extract_features(test_image_path)  # 입력 이미지에서 특징 벡터 추출
            lookalike, similarity = find_lookalike(db_path, test_features)  # 닮은꼴 연예인 찾기
            print(f"가장 닮은 연예인: {lookalike}")     # 닮은 연예인 출력
            print(f"유사도: {similarity:.2f}%")         # 유사도 출력
        else:
            print("유효하지 않은 이미지 경로입니다.")    # 유효하지 않은 경로 메시지 출력

if __name__ == '__main__':                            # 메인 함수 실행 조건
    main()                                            # 메인 함수 실행
