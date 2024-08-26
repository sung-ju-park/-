# 연예인 닮은꼴 찾기 프로젝트

이 프로젝트는 입력된 이미지의 얼굴 특징을 데이터베이스에 저장된 연예인들의 얼굴 특징과 비교하여 닮은 연예인을 찾는 기능을 제공합니다. 이 프로젝트는 주로 세 가지 단계로 이루어져 있습니다: 이미지 전처리(얼굴 탐지 및 크롭), VGG16 모델을 사용한 특징 추출, 그리고 닮은꼴 연예인 찾기입니다.

## 주요 기능

- **얼굴 탐지 및 전처리:** MTCNN을 사용하여 입력 이미지에서 얼굴을 탐지하고, 이를 잘라내어 전처리합니다.
- **특징 추출:** 사전 학습된 VGG16 모델을 활용하여 전처리된 이미지로부터 심층 특징 벡터를 추출합니다.
- **연예인 닮은꼴 매칭:** 입력 이미지의 특징과 데이터베이스에 저장된 연예인의 특징을 코사인 유사도를 통해 비교하여 가장 닮은 연예인을 찾습니다.
- **데이터베이스 관리:** SQLite를 사용하여 연예인 특징 벡터를 저장하고 조회합니다.

## 프로젝트 구조

- **`preprocess_images(input_dir, output_dir)`**: 입력된 디렉토리의 이미지를 얼굴 탐지 후 크롭하여 출력 디렉토리에 저장합니다.
- **`extract_features(img_path)`**: 지정된 이미지 파일에서 특징 벡터를 추출합니다.
- **`create_database()`**: SQLite 데이터베이스를 생성하고, 연예인의 특징 벡터를 저장할 테이블을 만듭니다.
- **`insert_features(db_path, celeb_name, feature_vector)`**: 연예인 이름과 특징 벡터를 데이터베이스에 삽입합니다.
- **`compute_similarity(feature1, feature2)`**: 두 특징 벡터 사이의 코사인 유사도를 계산합니다.
- **`find_lookalike(db_path, input_features)`**: 데이터베이스에서 입력된 특징 벡터와 가장 유사한 연예인을 찾습니다.

## 실행 방법

1. **데이터 전처리,특징 추출 및 데이터베이스 구축** 
   - `preprocess_images(input_dir, output_dir)` 함수를 사용하여 입력 디렉토리(`input_dir`)의 모든 이미지를 전처리하여 출력 디렉토리(`output_dir`)에 저장합니다.
   - `main()` 함수를 실행하여 전처리된 이미지에서 특징 벡터를 추출하고, 이를 SQLite 데이터베이스에 저장합니다.

  ```
   ---- python dd.py 실행
  ```

2. **테스트:**
   - `main()` 함수 실행 중, 테스트 이미지의 경로를 입력하면 닮은꼴 연예인을 찾아줍니다.
   ```
   --- python tt.py
   ```

## 요구사항

- Python 3.x
- 필수 라이브러리:
  - `numpy`
  - `tensorflow`
  - `Pillow`
  - `mtcnn`
  - `scipy`
  - `sqlite3`
  - `tqdm`

## 참고사항

이 코드는 교육 목적과 개인 프로젝트용으로 작성되었으며, 상업적 사용을 권장하지 않습니다. 각 연예인의 얼굴 이미지는 해당 저작권자의 권리에 속할 수 있습니다.





<details>
<summary># AutoCrawler</summary>
Google, Naver multiprocess image crawler (High Quality & Speed & Customizable)

![](docs/animation.gif)

# How to use

1. Install Chrome

2. pip install -r requirements.txt

3. Write search keywords in keywords.txt

4. **Run "main.py"**

5. Files will be downloaded to 'download' directory.


# Arguments
usage:
```
python3 main.py [--skip true] [--threads 4] [--google true] [--naver true] [--full false] [--face false] [--no_gui auto] [--limit 0]
```

```
--skip true        Skips keyword if downloaded directory already exists. This is needed when re-downloading.

--threads 4        Number of threads to download.

--google true      Download from google.com (boolean)

--naver true       Download from naver.com (boolean)

--full false       Download full resolution image instead of thumbnails (slow)

--face false       Face search mode

--no_gui auto      No GUI mode. (headless mode) Acceleration for full_resolution mode, but unstable on thumbnail mode.
                   Default: "auto" - false if full=false, true if full=true
                   (can be used for docker linux system)
                   
--limit 0          Maximum count of images to download per site. (0: infinite)
--proxy-list ''    The comma separated proxy list like: "socks://127.0.0.1:1080,http://127.0.0.1:1081".
                   Every thread will randomly choose one from the list.
```


# Full Resolution Mode

You can download full resolution image of JPG, GIF, PNG files by specifying --full true

![](docs/full.gif)



# Data Imbalance Detection

Detects data imbalance based on number of files.

When crawling ends, the message show you what directory has under 50% of average files.

I recommend you to remove those directories and re-download.


# Remote crawling through SSH on your server

```
sudo apt-get install xvfb <- This is virtual display

sudo apt-get install screen <- This will allow you to close SSH terminal while running.

screen -S s1

Xvfb :99 -ac & DISPLAY=:99 python3 main.py
```

# Customize

You can make your own crawler by changing collect_links.py

# How to fix issues

As google site consistently changes, you may need to fix ```collect_links.py```

1. Go to google image. [https://www.google.com/search?q=dog&source=lnms&tbm=isch](https://www.google.com/search?q=dog&source=lnms&tbm=isch)
2. Open devloper tools on Chrome. (CTRL+SHIFT+I, CMD+OPTION+I)
3. Designate an image to capture.
![CleanShot 2023-10-24 at 17 59 57@2x](https://github.com/YoongiKim/AutoCrawler/assets/38288705/6488d6df-1f01-4dfd-8691-6c0ac142fc04)
4. Checkout collect_links.py
![CleanShot 2023-10-24 at 18 02 35@2x](https://github.com/YoongiKim/AutoCrawler/assets/38288705/097c6c03-dd43-45d4-939e-2f677f595362)
5. Docs for XPATH usage: [https://www.w3schools.com/xml/xpath_syntax.asp](https://www.w3schools.com/xml/xpath_syntax.asp)
6. You can test XPATH using CTRL+F on your chrome developer tools.
![CleanShot 2023-10-24 at 18 05 14@2x](https://github.com/YoongiKim/AutoCrawler/assets/38288705/7ce2601f-9d53-48ff-a1cf-1a2befcc510f)
7. You need to find logic to crawling to work.

</details> 




