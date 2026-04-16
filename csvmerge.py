import pandas as pd

# 파일 로드
df1 = pd.read_csv("data_channel_extended_processed.csv")
df2 = pd.read_csv("matched_url_label.csv")
df2 = df2.dropna(subset=["label"])
label_map = {
    "Progressive":  0,
    "Neutral":      1,
    "Conservative": 2,
}

# url 기준으로 병합
df2["label"] = df2["label"].map(label_map)
print(df2["label"].isna().sum())  # map 후에 확인
merged = pd.merge(df1, df2, on="url", how="inner")  # how 옵션 선택 가능


# 결과 저장
merged.to_csv("data_channel.csv", index=False)

print(f"df1: {len(df1)}행, df2: {len(df2)}행 → 병합 결과: {len(merged)}행")