import os                                      # 운영 체제와 상호작용하기 위한 모듈을 불러옵니다.
import numpy as np                             # 수치 연산을 위한 파이썬 라이브러리를 불러옵니다.
import sqlite3                                 # SQLite 데이터베이스와 상호작용하기 위한 모듈을 불러옵니다.
from mtcnn import MTCNN                        # 얼굴을 탐지하기 위해 MTCNN 모델을 불러옵니다.
from PIL import Image                          # 이미지 처리를 위한 PIL 라이브러리를 불러옵니다.
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # VGG16 모델과 입력 전처리 함수를 불러옵니다.
from tensorflow.keras.preprocessing import image # 이미지를 처리하기 위한 Keras 모듈을 불러옵니다.
from scipy.spatial.distance import cosine      # 코사인 유사도를 계산하기 위한 함수를 불러옵니다.

# 얼굴 검출 및 전처리 함수
def preprocess_image(img_path):
    detector = MTCNN()                         # MTCNN 얼굴 탐지기 초기화
    img = Image.open(img_path).convert('RGB')  # 이미지를 열어 RGB 형식으로 변환
    img_array = np.array(img)                  # 이미지를 NumPy 배열로 변환
    faces = detector.detect_faces(img_array)   # 얼굴을 탐지
    if faces:                                  # 얼굴이 감지되면
        x, y, w, h = faces[0]['box']           # 첫 번째 얼굴의 위치와 크기 추출
        face = img.crop((x, y, x+w, y+h))      # 얼굴 영역을 잘라내기
        face = face.resize((224, 224))         # 얼굴 이미지를 224x224 크기로 리사이즈
        return face                            # 전처리된 얼굴 이미지 반환
    else:
        print(f"얼굴을 검출할 수 없습니다: {img_path}")  # 얼굴이 감지되지 않으면 경고 메시지 출력
        return img.resize((224, 224))          # 얼굴이 없을 경우 원본 이미지를 리사이즈하여 반환

# 얼굴 특징 추출 함수
def extract_features(img):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # VGG16 모델 로드 및 설정
    img_array = image.img_to_array(img)        # 이미지를 배열로 변환
    expanded_img = np.expand_dims(img_array, axis=0)  # 차원을 추가하여 배치 형태로 변환
    preprocessed_img = preprocess_input(expanded_img)  # VGG16 모델에 맞게 이미지를 전처리
    features = model.predict(preprocessed_img)  # 모델을 통해 특징 벡터 추출
    return features.flatten()                  # 추출된 특징 벡터를 1차원 배열로 반환

# 유사도 비교 알고리즘 함수
def compute_similarity(feature1, feature2):
    similarity = 1 - cosine(feature1, feature2)  # 코사인 유사도를 계산하여 두 벡터 간의 유사도 계산
    return max(0, min(similarity, 1))           # 유사도를 0과 1 사이의 값으로 제한하여 반환

# 특징 벡터의 크기를 조정하는 함수
def adjust_vector_size(vector, target_size):
    if len(vector) > target_size:               # 벡터가 목표 크기보다 크다면
        return vector[:target_size]             # 벡터를 목표 크기로 자름
    elif len(vector) < target_size:             # 벡터가 목표 크기보다 작다면
        return np.pad(vector, (0, target_size - len(vector)), 'constant')  # 0으로 채워 목표 크기로 만듦
    else:
        return vector                           # 벡터 크기가 이미 목표 크기와 같다면 그대로 반환

# 연예인 닮은꼴 찾기 함수
def find_lookalike(db_path, input_features):
    conn = sqlite3.connect(db_path)             # 데이터베이스에 연결
    c = conn.cursor()                           # 커서 객체 생성
    c.execute("SELECT celeb_name, feature_vector FROM features")  # 모든 연예인의 이름과 특징 벡터를 가져옴
    results = c.fetchall()                      # 쿼리 결과를 가져옴
    conn.close()                                # 데이터베이스 연결 종료

    max_similarity = 0                          # 초기 최대 유사도를 0으로 설정
    lookalike = None                            # 초기 닮은꼴 연예인을 None으로 설정
    for celeb_name, feature_bytes in results:   # 각 연예인의 특징 벡터와 비교
        db_features = np.frombuffer(feature_bytes, dtype=np.float32)  # 데이터베이스에서 가져온 벡터를 NumPy 배열로 변환
        
        if len(db_features) != len(input_features):  # 벡터의 크기가 다를 경우
            target_size = min(len(db_features), len(input_features))  # 두 벡터의 최소 크기 계산
            db_features = adjust_vector_size(db_features, target_size)  # 벡터 크기 조정
            input_features_adjusted = adjust_vector_size(input_features, target_size)  # 입력 벡터 크기 조정
        else:
            input_features_adjusted = input_features  # 크기가 같으면 조정 없이 그대로 사용
        
        similarity = compute_similarity(input_features_adjusted, db_features)  # 두 벡터 간 유사도 계산
        if similarity > max_similarity:        # 계산된 유사도가 최대 유사도보다 큰 경우
            max_similarity = similarity        # 최대 유사도 갱신
            lookalike = celeb_name             # 닮은꼴 연예인 갱신

    return lookalike, max_similarity * 100     # 최종적으로 가장 닮은 연예인과 유사도 반환

# 메인 실행 함수
def main():
    db_path = os.path.join(os.path.expanduser('~'), 'celebrity_features.db')  # 사용자 홈 디렉토리에서 데이터베이스 파일 경로 설정
    
    if not os.path.exists(db_path):            # 데이터베이스 파일이 존재하지 않을 경우
        print("오류: 데이터베이스 파일이 존재하지 않습니다.")  # 오류 메시지 출력
        return

    print("기존 데이터베이스를 사용합니다.")  # 데이터베이스가 존재하면 이를 사용하겠다고 알림

    while True:
        print("\n테스트 시작...")              # 테스트 시작 메시지 출력
        test_image_path = input("테스트할 이미지 경로를 입력하세요 (종료하려면 'q' 입력): ")  # 사용자에게 이미지 경로 입력 요청
        if test_image_path.lower() == 'q':     # 사용자가 'q'를 입력하면
            break                              # 프로그램 종료
        if os.path.isfile(test_image_path):    # 입력 경로가 파일인지 확인
            preprocessed_img = preprocess_image(test_image_path)  # 이미지를 전처리
            test_features = extract_features(preprocessed_img)  # 전처리된 이미지에서 특징 벡터 추출
            lookalike, similarity = find_lookalike(db_path, test_features)  # 가장 닮은 연예인을 찾음
            print(f"가장 닮은 연예인: {lookalike}")  # 닮은 연예인의 이름 출력
            print(f"유사도: {similarity:.2f}%")       # 유사도를 백분율로 출력
        elif os.path.isdir(test_image_path):   # 입력 경로가 디렉토리일 경우
            print("입력한 경로는 디렉토리입니다. 이미지 파일의 경로를 입력해주세요.")  # 경고 메시지 출력
        else:
            print("유효하지 않은 이미지 경로입니다.")  # 잘못된 경로가 입력되었을 때 경고 메시지 출력

if __name__ == '__main__':                     # 메인 함수 실행 조건 확인
    main()                                     # 메인 함수 실행
