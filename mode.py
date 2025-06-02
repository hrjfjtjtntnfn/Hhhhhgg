import requests
import pandas as pd
import numpy as np

# Thư viện cho mô hình
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def fetch_history(size=100):
    """
    Gọi API để lấy lịch sử `size` phiên gần nhất, trả về danh sách dict:
    [
      {"d1": x, "d2": y, "d3": z, "tola": sum, "sid": "12345", "timeMilli": 1678901234567},
      ...
    ]
    """
    label = []
    url = "https://api.wsktnus8.net/v2/history/getLastResult"
    params = {
        "gameId": "ktrng_3979",
        "size": size,
        "tableId": "39791215743193",
        "curPage": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    for i in data['data']['resultList']:
        label.append({
            "d1": i['facesList'][0],          # Xúc xắc 1
            "d2": i['facesList'][1],          # Xúc xắc 2
            "d3": i['facesList'][2],          # Xúc xắc 3
            "tola": i['score'],               # Tổng điểm
            "sid": int(i['gameNum'].lstrip("#")),  # Mã phiên (đã lstrip “#” và chuyển thành int)
            "timeMilli": i['timeMilli']       # Thời gian mở phiên (millisecond)
        })
    return label

def prepare_dataframe(history_list):
    """
    Chuyển danh sách dict thành DataFrame, sắp xếp theo sid, tạo thêm cột ngày giờ,
    rồi tạo mục tiêu `tola_next` chính là tổng của phiên kế tiếp.
    """
    # 1) Tạo DataFrame từ list
    df = pd.DataFrame(history_list)

    # 2) Chuyển `timeMilli` sang datetime (tính theo UTC). 
    #    Nếu muốn múi giờ Asia/Bangkok, có thể thêm tz_localize/tz_convert.
    df['dt'] = pd.to_datetime(df['timeMilli'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')

    # 3) Sắp xếp theo sid tăng dần (phiên cũ -> mới).
    df = df.sort_values('sid').reset_index(drop=True)

    # 4) Tách thành các features thời gian: giờ, phút, giây, ngày trong tuần …
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['second'] = df['dt'].dt.second
    df['weekday'] = df['dt'].dt.weekday  # thứ 0=Thứ Hai, … 6=Chủ nhật

    # 5) Tạo cột mục tiêu `tola_next`: shift(-1) trên cột `tola`
    #    (phiên hiện tại dùng để dự đoán phiên kế tiếp)
    df['tola_next'] = df['tola'].shift(-1)

    # 6) Loại bỏ dòng cuối cùng vì không có `tola_next`
    df = df.iloc[:-1].copy()

    return df

def train_and_predict_top4(df):
    """
    Huấn luyện RandomForestClassifier để dự đoán `tola_next` (tổng của phiên kế tiếp).
    Trả về:
      - model: mô hình đã huấn luyện
      - X_last: features của phiên cuối cùng trong df (để dự đoán phiên tiếp theo)
      - proba: xác suất dự đoán cho từng tổng (classes_)
      - top4: list 4 tổng có xác suất cao nhất
    """
    # 1) Chọn features và target
    feature_cols = ['tola', 'hour', 'minute', 'second', 'weekday']
    X = df[feature_cols]
    y = df['tola_next']              # các giá trị từ 3 đến 18

    # 2) Vì dữ liệu khá nhỏ (n≈99), ta có thể dùng toàn bộ để train; 
    #    nếu cần test thì chia train/test theo thứ tự thời gian. Ở đây ta train trên toàn bộ để dự đoán 1 phiên tương lai.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    # 3) Lấy features của phiên “cuối cùng” (mới nhất trong df) để dự đoán phiên kế tiếp
    X_last = df.iloc[[-1]][feature_cols]

    # 4) Dự đoán xác suất cho từng giá trị tổng có thể (model.classes_)
    proba = model.predict_proba(X_last)[0]  # 1D array với xác suất cho từng class

    # 5) Ghép class và probability, chọn top 4
    classes = model.classes_                 # ví dụ array([3,4,5,...,18])
    df_proba = pd.DataFrame({
        'total': classes,
        'proba': proba
    })
    # Sắp xếp giảm dần theo xác suất, lấy 4 đầu
    df_proba = df_proba.sort_values('proba', ascending=False).reset_index(drop=True)
    top4 = df_proba.head(4)[['total', 'proba']]

    return model, X_last, df_proba, top4

if __name__ == "__main__":
    # 1) Lấy 100 phiên gần nhất
    hist = fetch_history(size=100)

    # 2) Chuẩn hóa, tạo DataFrame
    df_hist = prepare_dataframe(hist)

    # 3) Huấn luyện và dự đoán
    model, X_last, df_proba_all, top4 = train_and_predict_top4(df_hist)

    # 4) In kết quả top 4 tổng điểm “phiên tiếp theo” có xác suất cao nhất
    print("Top 4 tổng điểm dự đoán cho phiên tiếp theo (có xác suất cao nhất):")
    print(top4.to_string(index=False, formatters={'proba': '{:.4f}'.format}))

    # Nếu muốn xem đầy đủ xác suất cho mọi tổng từ 3-18, có thể in df_proba_all:
    # print("\nXác suất cho mọi tổng (3..18):")
    # print(df_proba_all.to_string(index=False, formatters={'proba': '{:.4f}'.format}))
    