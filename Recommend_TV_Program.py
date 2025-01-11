from difflib import SequenceMatcher
from typing import List, Tuple
import pandas as pd
from jamo import h2j
from datetime import datetime, timedelta
import pytz
import openai
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# OpenAI API 키 설정
openai.api_key = "your_apikey"

def calculate_similarity(input_text: str, comparison_text: str) -> float:
    # Convert both inputs to strings to prevent errors
    input_text = str(input_text)
    comparison_text = str(comparison_text)

    # Convert both strings to Jamo (Hangul decomposition)
    input_jamo = ''.join(h2j(input_text))
    comparison_jamo = ''.join(h2j(comparison_text))

    # Use SequenceMatcher on decomposed Jamo strings
    return SequenceMatcher(None, input_jamo, comparison_jamo).ratio()

def find_closest_program(input_text: str, program_list: List[Tuple[str, str]]) -> List[str]:
    abbreviation_matches = []
    full_name_matches = []

    for full_name, abbreviation in program_list:
        # Ensure both values are strings
        full_name = str(full_name)
        abbreviation = str(abbreviation)

        # Calculate similarity for abbreviation and full name
        abbreviation_similarity = calculate_similarity(input_text, abbreviation)
        full_name_similarity = calculate_similarity(input_text, full_name)

        if abbreviation_similarity >= 0.7:
            abbreviation_matches.append((abbreviation_similarity, full_name))
        else:
            full_name_matches.append((full_name_similarity, full_name))

    if abbreviation_matches:
        abbreviation_matches.sort(reverse=True, key=lambda x: x[0])
        highest_similarity = abbreviation_matches[0][0]
        return [match[1] for match in abbreviation_matches if match[0] == highest_similarity]

    full_name_matches.sort(reverse=True, key=lambda x: x[0])
    if full_name_matches:
        highest_similarity = full_name_matches[0][0]
        return [match[1] for match in full_name_matches if match[0] == highest_similarity]

    return []

def load_program_list(file_path: str) -> List[Tuple[str, str]]:
    # Load data from Excel and ensure all columns are strings
    df = pd.read_excel(file_path)
    df = df.astype(str)  # Convert all columns to strings to avoid float errors
    program_list = list(zip(df['프로그램명'], df['줄임말']))
    return program_list

def load_schedule(schedule_path: str) -> pd.DataFrame:
    # Load schedule data
    schedule_df = pd.read_csv(schedule_path)
    # print("원본 스케줄 데이터프레임:")
    # print(schedule_df.head())

    # Time 컬럼을 HH:MM 형식의 문자열로 변환 (초 제거)
    schedule_df['Time'] = pd.to_datetime(schedule_df['Time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M')
    # print("Time 컬럼 변환 후 데이터프레임:")
    # print(schedule_df.head())

    return schedule_df

def filter_programs_by_time(current_time: datetime, schedule_df: pd.DataFrame) -> pd.DataFrame:
    # 복사본 생성
    schedule_df = schedule_df.copy()

    # StartTime 컬럼 생성 (datetime 객체로 변환)
    schedule_df['StartTime'] = schedule_df['Time'].apply(
        lambda t: datetime.strptime(t, "%H:%M").replace(
            year=current_time.year, month=current_time.month, day=current_time.day
        )
    )

    # StartTime이 datetime인지 확인
    if not pd.api.types.is_datetime64_any_dtype(schedule_df['StartTime']):
        schedule_df['StartTime'] = pd.to_datetime(schedule_df['StartTime'])

    # 현재 시간을 timezone-naive로 변환
    current_time_naive = current_time.replace(tzinfo=None)

    # TimeDiff 컬럼 생성 (현재 시간과 StartTime의 차이 계산)
    schedule_df['TimeDiff'] = (current_time_naive - schedule_df['StartTime']).dt.total_seconds() / 60

    # TimeDiff가 0 이상이고 30 이하인 행만 필터링
    schedule_df = schedule_df[(schedule_df['TimeDiff'] >= 0) & (schedule_df['TimeDiff'] <= 30)]

    # Time 컬럼에서 초 제거
    schedule_df['Time'] = schedule_df['StartTime'].dt.strftime('%H:%M')

    # 필요한 컬럼만 반환
    return schedule_df[['Time', 'Program', 'Channel']]


def find_similar_programs(program_name: str, schedule_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate similarity for each program in the schedule
    schedule_df['Similarity'] = schedule_df['Program'].apply(lambda x: calculate_similarity(program_name, x))
    # Filter programs with similarity >= 0.7
    similar_programs = schedule_df[schedule_df['Similarity'] >= 0.7]
    return similar_programs

def find_related_by_genre(program_name: str, program_list_df: pd.DataFrame) -> pd.DataFrame:
    # Find the genre and sub-genre of the given program name
    target_program = program_list_df[program_list_df['프로그램명'] == program_name]
    if not target_program.empty:
        genre = target_program.iloc[0]['장르']
        sub_genre = target_program.iloc[0]['장르 소분류']
        # Find all programs with the same genre and sub-genre
        related_programs = program_list_df[(program_list_df['장르'] == genre) & (program_list_df['장르 소분류'] == sub_genre) & (program_list_df['프로그램명'] != program_name)]
        return related_programs
    return pd.DataFrame()

def recommend_genre(input_text: str) -> str:
    # 장르 목록
    genres = [
        "게임", "교육", "낚시", "뉴스", "다큐", "라디오", "종교", "스포츠", "시니어",
        "예능", "애니", "어린이", "영화", "음악", "홈쇼핑", "드라마", "패션"
    ]

    # 조건에 따른 장르 매핑
    if any(keyword in input_text for keyword in ["게임", "프로게임", "프로그래머"]):
        return "게임"
    elif any(keyword in input_text for keyword in ["드라마", "우울", "슬퍼", "힘들어"]):
        return "드라마"
    elif any(keyword in input_text for keyword in ["음악", "위로", "안정", "편안"]):
        return "음악"
    elif any(keyword in input_text for keyword in ["예능", "심심", "재밌게", "웃겨", "바보", "웃긴"]):
        return "예능"
    elif any(keyword in input_text for keyword in ["교육", "공부", "배우고", "학습"]):
        return "교육"
    elif any(keyword in input_text for keyword in ["자연", "낚시", "고요"]):
        return "낚시"
    elif any(keyword in input_text for keyword in ["시사", "뉴스", "정치"]):
        return "뉴스"
    elif any(keyword in input_text for keyword in ["라디오", "청취", "소리"]):
        return "라디오"
    elif any(keyword in input_text for keyword in ["패션", "옷", "스타일"]):
        return "패션"
    elif any(keyword in input_text for keyword in ["운동", "스포츠", "땀"]):
        return "스포츠"
    elif any(keyword in input_text for keyword in ["종교", "기도", "신앙"]):
        return "종교"
    elif any(keyword in input_text for keyword in ["아이", "어린이", "애기", "아기", "유아"]):
        return "어린이"
    elif any(keyword in input_text for keyword in ["홈쇼핑", "구매", "쇼핑"]):
        return "홈쇼핑"
    elif any(keyword in input_text for keyword in ["영화", "시상식", "소개"]):
        return "영화"
    elif any(keyword in input_text for keyword in ["다큐", "기록", "진지"]):
        return "다큐"
    elif any(keyword in input_text for keyword in ["시니어", "노인", "어르신"]):
        return "시니어"
    else:
        # 조건에 맞는 키워드가 없으면 랜덤으로 추천
        return random.choice(genres)

def recommend_from_cluster_and_genre_with_airing(
    cluster_id: int, 
    recommended_genre: str, 
    program_preferences: pd.DataFrame, 
    program_list_df: pd.DataFrame, 
    schedule_df: pd.DataFrame, 
    current_time_kst: datetime
):
    # 클러스터에서 데이터 필터링
    cluster_programs = program_preferences[program_preferences['Cluster'] == cluster_id]
    if cluster_programs.empty:
        print(f"클러스터 {cluster_id}에 해당하는 데이터가 없습니다.")
        return
    
    # 추천 장르와 일치하는 프로그램 필터링
    genre_programs = program_list_df[program_list_df['장르'] == recommended_genre]
    cluster_genre_programs = cluster_programs[cluster_programs['Program Name'].isin(genre_programs['프로그램명'])]
    if cluster_genre_programs.empty:
        print(f"[추천 장르: {recommended_genre}] 클러스터 {cluster_id}에서 추천할 프로그램이 없습니다.")
        return
    
    # 상위 3개 프로그램 선택
    cluster_genre_programs = cluster_genre_programs.sort_values(by='Proportion', ascending=False).head(3)
    
    print(f"\n[추천 장르: {recommended_genre} | 클러스터 {cluster_id}의 추천 프로그램]")
    for _, row in cluster_genre_programs.iterrows():
        program_name = row['Program Name']
        airing_schedule = find_similar_programs(program_name, schedule_df)
        filtered_schedule = filter_programs_by_time(current_time_kst, airing_schedule)

        if not filtered_schedule.empty:
            print(f"{program_name} (선호도 비율: {row['Proportion']:.2f})")
            print(filtered_schedule[['Time', 'Program', 'Channel']].to_string(index=False))
        else:
            print(f"{program_name} (선호도 비율: {row['Proportion']:.2f}) - 현재 방영 중인 프로그램이 없습니다.")

    # 추천 장르에 맞는 프로그램 중 현재 방영 중인 프로그램 확인
    genre_schedule = schedule_df[schedule_df['Program'].isin(
        genre_programs['프로그램명']
    )]
    filtered_genre_schedule = filter_programs_by_time(current_time_kst, genre_schedule)
    if not filtered_genre_schedule.empty:
        print(f"\n[{recommended_genre} 장르의 현재 방영 중인 프로그램]")
        print(filtered_genre_schedule[['Time', 'Program', 'Channel']].to_string(index=False))
    else:
        print(f"[{recommended_genre}] 장르의 현재 방영 중인 프로그램이 없습니다.")

        
def process_input(input_text: str, program_list: List[Tuple[str, str]], schedule_df: pd.DataFrame, program_list_df: pd.DataFrame, program_preferences: pd.DataFrame):
    # 한국 표준시 기준으로 현재 시간 출력
    kst = pytz.timezone('Asia/Seoul')
    current_time_kst = datetime.now(kst)
    current_time_str = current_time_kst.strftime("%H:%M")
    print(f"현재 시간 (한국 표준시): {current_time_str}")

    # 키워드 포함 여부 확인
    contains_keywords = any(keyword in input_text for keyword in ["비슷", "유사", "관련", "같은"])

    # 키워드 제거_프로그램명만 추출하기 위해
    for keyword in ["비슷", "유사", "관련", "같은"]:
        input_text = input_text.replace(keyword, "").strip()

    # 가장 유사한 프로그램 찾기
    closest_programs = find_closest_program(input_text, program_list)

    valid_programs = []
    for program_name in closest_programs:
        similarity = calculate_similarity(input_text, program_name)
        if similarity >= 0.52:
            valid_programs.append(program_name)
            
    # 키워드 포함된 경우: 장르 기반 추천 (입력된 프로그램 제외)
    if contains_keywords and valid_programs:
        program_name = valid_programs[0]  # 가장 유사한 프로그램 선택
        target_program = program_list_df[program_list_df['프로그램명'] == program_name]

        if not target_program.empty:
            genre = target_program.iloc[0]['장르']
            print(f"프로그램 이름: {program_name}, 장르: {genre}")

            # 동일 장르의 현재 방영 중인 프로그램 추천 (입력된 프로그램 제외)
            genre_schedule = schedule_df[
                schedule_df['Program'].isin(
                    program_list_df[program_list_df['장르'] == genre]['프로그램명']
                )
            ]
            genre_schedule = genre_schedule[genre_schedule['Program'] != program_name]  # 입력된 프로그램 제외
            filtered_genre_schedule = filter_programs_by_time(current_time_kst, genre_schedule)

            if not filtered_genre_schedule.empty:
                print(f"\n[{genre} 장르의 현재 방영 중인 프로그램]")
                print(filtered_genre_schedule[['Time', 'Program', 'Channel']].to_string(index=False))
            else:
                print(f"[{genre}] 장르의 현재 방영 중인 프로그램이 없습니다.")
        return  # 장르 기반 추천 후 종료

    # 유사한 프로그램 정보 출력
    if valid_programs:
        print("프로그램 이름 및 정보:")
        for program_name in valid_programs:
            target_program = program_list_df[program_list_df['프로그램명'] == program_name]
            if not target_program.empty:
                genre = target_program.iloc[0]['장르']
                sub_genre = target_program.iloc[0]['장르 소분류']
                similarity = calculate_similarity(input_text, program_name)
                print(f"{program_name} (유사도: {similarity * 100:.2f}%, 장르: {genre}, 장르 소분류: {sub_genre})")

                # 현재 방영 중인지 확인
                schedule_matches = find_similar_programs(program_name, schedule_df)
                filtered_schedule = filter_programs_by_time(current_time_kst, schedule_matches)
                if not filtered_schedule.empty:
                    print(f"\n[{program_name} 관련 편성표]")
                    print(filtered_schedule[['Time', 'Program', 'Channel']].to_string(index=False))
                else:
                    print(f"[{program_name}] 현재 방영 중인 프로그램이 없습니다.")

                    # 같은 장르에서 추천
                    genre_schedule = schedule_df[schedule_df['Program'].isin(
                        program_list_df[program_list_df['장르'] == genre]['프로그램명']
                    )]
                    filtered_genre_schedule = filter_programs_by_time(current_time_kst, genre_schedule)
                    if not filtered_genre_schedule.empty:
                        print(f"\n[{genre} 장르의 현재 방영 중인 프로그램]")
                        print(filtered_genre_schedule[['Time', 'Program', 'Channel']].to_string(index=False))
                    else:
                        print("현재 방영 중인 동일 장르 프로그램이 없습니다.")
            else:
                print(f"{program_name} (정보를 찾을 수 없습니다)")
    else:
            # 유사도 0.8 이상의 프로그램을 찾지 못한 경우 처리
            print("유사도 0.8 이상의 프로그램을 찾지 못했습니다. 독백체로 감지되었습니다.")
    
            # 독백체일 경우 장르 추천 
            recommended_genre = recommend_genre(input_text)
            print(f"추천 장르: {recommended_genre}")

            # 랜덤 클러스터 할당
            random_cluster = random.randint(0, program_preferences['Cluster'].max())
            print(f"랜덤 클러스터: {random_cluster}")
            recommend_from_cluster_and_genre_with_airing(random_cluster, recommended_genre, program_preferences, program_list_df, schedule_df, current_time_kst)

# File paths
file_path = "your_filepath"
schedule_path = "your_schedulepath"

# Load the program list and schedule
program_list_df = pd.read_excel(file_path).fillna("").astype(str)  # Load full program list for genre matching
schedule_df = load_schedule(schedule_path)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load user data (e.g., voice data with program name, mean pitch, etc.)
data_path = "your_datapath"
data = pd.read_csv(data_path)

# 클러스터링에 사용할 특징 컬럼 선택
features = data[['Mean Pitch', 'Voiced Duration / Total Duration']]

# 데이터 스케일링
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans 모델 생성 및 클러스터링 수행
optimal_k = 3  # Elbow Method 등을 통해 결정된 최적 K값
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)  # 클러스터 결과 추가

# 클러스터별 프로그램 선호도 집계
program_preferences = (
    data.groupby(['Cluster', 'Program Name'])
    .size()
    .reset_index(name='Count')
)
program_preferences['Proportion'] = (
    program_preferences.groupby('Cluster')['Count']
    .transform(lambda x: (x / x.sum() if x.sum() > 0 else 0) * 100)
)


# 클러스터별 상위 3개 프로그램 추출
top_programs = (
    program_preferences
    .sort_values(['Cluster', 'Proportion'], ascending=[True, False])
    .groupby('Cluster')
    .head(3)
    .reset_index(drop=True)
)

def process_cluster_input(cluster_id: int, program_preferences: pd.DataFrame):
    # 클러스터 ID에 해당하는 프로그램 필터링
    cluster_programs = program_preferences[program_preferences['Cluster'] == cluster_id]
    if cluster_programs.empty:
        print(f"클러스터 {cluster_id}에 해당하는 데이터가 없습니다.")
        return

    # 상위 3개 프로그램만 출력
    cluster_top_programs = (
        cluster_programs
        .sort_values(by='Proportion', ascending=False)
        .head(3)
    )

    print(f"\n클러스터 {cluster_id}에서 많이 본 상위 3개 프로그램:")
    for _, row in cluster_top_programs.iterrows():
        print(f"{row['Program Name']} (Count: {row['Count']}, Proportion: {row['Proportion']:.2f})")


# Input text
input_text = input("입력 텍스트를 입력하세요: ")

# Process the input
process_input(
    input_text, 
    program_list_df[['프로그램명', '줄임말']].values.tolist(), 
    schedule_df,
    program_list_df, 
    program_preferences  # 클러스터링 결과 전달
)