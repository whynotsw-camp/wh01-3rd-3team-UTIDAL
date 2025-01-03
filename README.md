## 📺 음성 검색 기반 IPTV 채널 및 컨텐츠 추천 서비스 🔉

---------------------------------------

# 프로젝트 계획서

## 1. 🎂 프로젝트 개요
- **프로젝트명**: 음성 검색 기반 IPTV 채널 및 컨텐츠 추천 서비스
- **목표** 
   1. 편성표 크롤링을 통한 편성표 데이터 취합
   2. 방송 프로그램 포스터 크롤링을 통한 포스터 데이터 수집
   3. 유사도 및 군집화 클러스터링을 통한 유저별 프로그램 추천
- **기간**: 2024년 6월 - 2025년 1월

## 2. ⏰ 프로젝트 일정
- **분석 및 설계**: 2025년 8월 19일 - 11월 14일
- **프로토 타입 개발**: 2025년 12월 14일 - 12월 17일
- **최종 프로젝트 개발 및 테스트**: 2024년 12월 20일 - 2024년 1월 3일

## 3. 💪 팀 구성(이름을 클릭하면 해당 인원의 Git으로 이동)
|이름|[박종현](https://github.com/JayParc)|[권정인](https://github.com/yojeong125)|[신새봄](https://github.com/SaebomSHIN)|[이중찬](https://github.com/Chan2a)|[정연진](https://github.com/yeonjin118)|
|---|:---:|:---:|:---:|:---:|:---:|
|역할|프론트&백엔드 개발|프로젝트 매니저|빅데이터, AI|인프라, 백엔드|빅데이터, AI|

## 4.📚 STACKS
* 주 개발 언어
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
* 웹 구현
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white)  ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=HTML5&logoColor=white)  ![Javascript](https://img.shields.io/badge/Javascript-F7DF1E?style=for-the-badge&logo=Javascript&logoColor=white)  ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=CSS3&logoColor=white)
* 데이터 크롤링  
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=Selenium&logoColor=white)  ![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup4-FFD700?style=for-the-badge&logo=BeautifulSoup&logoColor=black)
* 데이터 분석
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white)  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white)
* 클라우드
![Amazon AWS](https://img.shields.io/badge/Amazon%20AWS-232F3E?style=for-the-badge&logo=Amazon%20AWS&logoColor=white)

---------------------------------------

# 🏁 기획서
[IPTV 채널 추천 서비스 기획서 PDF 파일](https://github.com/whynotsw-camp/wh01-3rd-3team-UTIDAL/blob/main/Report/%5BKDT%5D%203%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%B5%E1%84%92%E1%85%AC%E1%86%A8%E1%84%89%E1%85%A5_%E1%84%8B%E1%85%AE%E1%84%90%E1%85%B5%E1%84%83%E1%85%A1%E1%86%AF.pdf)

---------------------------------------

# 📕 요구사항 정의서
[IPTV 채널 추천 서비스 요구사항 정의서 PDF 파일](https://github.com/whynotsw-camp/wh01-3rd-3team-UTIDAL/blob/main/Report/%5BKDT%5D%20IPTV%20%E1%84%8E%E1%85%A2%E1%84%82%E1%85%A5%E1%86%AF%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%A5%E1%86%AB%20%E1%84%89%E1%85%A5%E1%84%87%E1%85%B5%E1%84%89%E1%85%B3%203%E1%84%90%E1%85%B5%E1%86%B7_%E1%84%8B%E1%85%AD%E1%84%80%E1%85%AE%E1%84%89%E1%85%A1%E1%84%92%E1%85%A1%E1%86%BC%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%8B%E1%85%B4%E1%84%89%E1%85%A5.pdf)

----------------------------------------

# ✈️ WBS
[IPTV 채널 추천 서비스 WBS 엑셀 파일](https://github.com/whynotsw-camp/wh01-3rd-3team-UTIDAL/raw/refs/heads/main/Report/%5BKDT%5D%20IPTV%20%E1%84%8E%E1%85%A2%E1%84%82%E1%85%A5%E1%86%AF%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%A5%E1%86%AB%20%E1%84%89%E1%85%A5%E1%84%87%E1%85%B5%E1%84%89%E1%85%B3%203%E1%84%90%E1%85%B5%E1%86%B7_WBS.xlsx)


-----------------------------------------

# 📚 모델 정의서 및 성능 평가서
[IPTV 채널 추천 서비스 모델 정의서 및 성능평가서](https://github.com/whynotsw-camp/wh01-3rd-3team-UTIDAL/blob/main/Report/%5BKDT%5D%20IPTV%20%E1%84%8E%E1%85%A2%E1%84%82%E1%85%A5%E1%86%AF%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%A5%E1%86%AB%20%E1%84%89%E1%85%A5%E1%84%87%E1%85%B5%E1%84%89%E1%85%B3%203%E1%84%90%E1%85%B5%E1%86%B7_%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%8B%E1%85%B4%E1%84%89%E1%85%A5%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%91%E1%85%A7%E1%86%BC%E1%84%80%E1%85%A1%E1%84%89%E1%85%A5.pdf)

-----------------------------------------


## 📌 컨벤션 및 규칙

### 커밋 컨벤션

| 태그 이름        | 설명                                                                                                     | 타입 |
| ---------------- | -------------------------------------------------------------------------------------------------------- | ---- |
| Feat             | 새로운 기능을 추가할 경우                                                                                | 기능 |
| Fix              | 버그를 고친 경우                                                                                         | 기능 |
| Design           | CSS 등 사용자 UI 디자인 변경                                                                             | 기능 |
| !BREAKING CHANGE | 커다란 API 변경의 경우                                                                                   | 기능 |
| !HOTFIX          | 급하게 치명적인 버그를 고쳐야하는 경우                                                                   | 기능 |
| Style            | 코드 포맷 변경, 세미 콜론 누락, 오타 수정, 탭 사이즈 변경, 변수명 변경 등 코어 로직을 안건드는 변경 사항 | 개선 |
| Refactor         | 프로덕션 코드 리팩토링                                                                                   | 개선 |
| Comment          | 필요한 주석 추가 및 변경                                                                                 | 개선 |
| Docs             | 문서(Readme.md)를 수정한 경우                                                                            | 수정 |
| Rename           | 파일 혹은 폴더명을 수정하거나 옮기는 작업만인 경우                                                       | 수정 |
| Remove           | 파일을 삭제하는 작업만 수행한 경우                                                                       | 수정 |
| Test             | 테스트 추가, 테스트 리팩토링(프로덕션 코드 변경 X)                                                       | 빌드 |
| Chore            | 빌드 태스트 업데이트, 패키지 매니저를 설정하는 경우(프로덕션 코드 변경 X)                                | 빌드 |

ex) feat: login logic add

### ⭐️⭐️⭐️⭐️⭐️git 규칙⭐️⭐️⭐️⭐️⭐️

1. 작업 브랜치 분기는 무조건 main 브랜치에서 분기한 후 작업
2. 커밋 메세지 준수할 것.

