import pandas as pd
import ollama
import json
import os
from tqdm import tqdm
from httpx import Client, Timeout

def load_prompt(file_path):
    """프롬프트 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_json_from_llm(content):
    """LLM 응답에서 JSON만 추출하여 딕셔너리로 변환"""
    content = content.strip()
    if "```" in content:
        # 마크다운 블록 제거
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except Exception:
        # 파싱 실패 시 원본을 텍스트로 담아 반환
        return {"error": "JSON_PARSE_FAILED", "raw_content": content}

def run_automation():
    # 1. 경로 설정
    input_path = 'data_raw/data_channel_extended.csv'
    output_folder = 'video_summary'
    output_json_path = os.path.join(output_folder, 'analysis_results_v2.json')
    error_csv_path = os.path.join(output_folder, 'error_urls_v2.csv')
    
    # 프롬프트 파일 경로
    prompt_file1 = os.path.join(output_folder, 'prompt_summary1.txt')
    prompt_file2 = os.path.join(output_folder, 'prompt_summary2.txt')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 데이터 및 프롬프트 로드
    df = pd.read_csv(input_path)
    prompt_template1 = load_prompt(prompt_file1)
    prompt_template2 = load_prompt(prompt_file2)

    # 타임아웃 90초 설정
    custom_client = ollama.Client(timeout=Timeout(90.0))

    results = []
    error_urls = []

    print(f"🚀 총 {len(df)}개의 영상을 2단계 파이프라인으로 분석합니다...")

    # 3. 데이터 루프
    for index, row in tqdm(df.iterrows(), total=len(df)):
        url = str(row['url'])
        channel = str(row['channel_name'])
        title = str(row['title'])
        comments = str(row['comment'])

        try:
            # --- [STAGE 1: 제목 독립 분석] ---
            final_prompt1 = prompt_template1.replace("{channel}", channel).replace("{title}", title)
            
            step1_content=None
            res1 = custom_client.chat(
                model='gpt-oss',
                messages=[{'role': 'user', 'content': final_prompt1}],
                options={'temperature': 0}
            )
            step1_content = res1['message']['content']
            step1_result = parse_json_from_llm(step1_content)

            # --- [STAGE 2: 문맥 복원 및 여론 요약] ---
            # 1단계 결과를 JSON 문자열로 변환하여 2단계 프롬프트에 주입
            step1_json_str = json.dumps(step1_result, ensure_ascii=False)
            
            # 프롬프트 내의 {comments}와 1단계 결과(Input Context 대용)를 치환
            # prompt_summary2.txt 내용에 맞춰 변수명을 조절했습니다.
            final_prompt2 = prompt_template2.replace('{"intent": "INSUFFICIENT", "confidence_score": 0.2}', step1_json_str) # 예시 치환용
            # 만약 파일에 변수 태그 형식을 쓰고 싶다면 아래 방식을 권장합니다:
            final_prompt2 = prompt_template2.replace("{comments_lists}", comments).replace("{json_from_step1}", step1_json_str)
            
            step2_content=None
            res2 = custom_client.chat(
                model='gpt-oss',
                messages=[{'role': 'user', 'content': final_prompt2}],
                options={'temperature': 0}
            )
            
            step2_content = res2['message']['content']
            step2_result = parse_json_from_llm(step2_content)

            # 최종 결과 저장 구조
            processed_data = {
                "url": url,
                "channel": channel,
                "title": title,
                "stage1": step1_result,
                "stage2": step2_result
            }
            results.append(processed_data)

            # 매 루프마다 결과 저장 (안전장치)
            with open(output_json_path, 'w', encoding='utf-8') as jf:
                json.dump(results, jf, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\n❌ {index}번 행(URL: {url}) 오류 발생: {e}")
            error_urls.append({
                "url": url, 
                "index": index, 
                "error_msg": str(e),
                "ans1": step1_content or "Didn't Answered",
                "ans2": step2_content or "Didn't Answered"
            })
            error_df = pd.DataFrame(error_urls)
            error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 분석 완료! 결과가 {output_json_path}에 저장되었습니다.")

if __name__ == "__main__":
    run_automation()