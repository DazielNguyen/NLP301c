# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 2: Sentiment Analysis with Naive Bayes
### Probability and Bayes' Rule

> Video này cung cấp cái nhìn tổng quan về **Quy tắc Bayes** (Bayes' rule) và **xác suất** (probability).

![01_Probability_and_Bayes'_Rule](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/01_Probability_and_Bayes'_Rule.png)

- Xác suất là cơ bản cho nhiều ứng dụng trong **NLP**, ví dụ như phân loại tweet (tích cực hay tiêu cực).
    1. Xem xét **xác suất** và **xác suất có điều kiện** (conditional probability).
    2. Rút ra **Quy tắc Bayes** từ xác suất có điều kiện.
- Quy tắc Bayes được áp dụng rộng rãi (y học, giáo dục, **NLP**).
- Bạn sẽ dùng nó để thực hiện **phân tích tình cảm** (sentiment analysis) (nhiệm vụ tuần này).
- Trong khóa học tiếp theo, bạn cũng sẽ dùng nó để **tự động sửa** (auto-correct).
> Ví dụ: Một kho (corpus) lớn các tweet được phân loại là **tích cực** hoặc **tiêu cực**, nhưng **không phải cả hai**. Từ "happy" (hạnh phúc) có thể xuất hiện ở cả hai loại.

![02_Probability_and_Bayes'_Rule_Example](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/02_Probability_and_Bayes'_Rule_Example.png)

- Một cách nghĩ về xác suất là đếm **tần suất** (frequencies).
- **Sự kiện A** (Event A): Tweet được gắn nhãn tích cực.
- **Xác suất của A** ($P(A)$) = (Số tweet tích cực) / (Tổng số tweet).
- Ví dụ: $13/20 = 0.65$ (hoặc 65%).
- **Xác suất bổ sung** (complementary probability): $P(\text{tiêu cực}) = 1 - P(\text{tích cực})$. (Điều này đúng vì các tweet chỉ được phân loại là tích cực hoặc tiêu cực, không phải cả hai).
- **Sự kiện B** (Event B): Tweet chứa từ "happy" (hạnh phúc). Ví dụ: $N_{\text{happy}} = 4$.
- **Giao điểm** (intersection): Các tweet được gắn nhãn tích cực VÀ chứa từ "happy".
- Xác suất của giao điểm = (Diện tích giao điểm) / (Diện tích toàn bộ corpus).
- Ví dụ: 3 tweet (tích cực VÀ "happy") trên tổng số 20 tweet $\rightarrow$ Xác suất = $3/20 = 0.15$.
- Video tiếp theo sẽ nói về **Naive Bayes**.
---
### Bayes' Rule
- Video này xem xét **xác suất có điều kiện** (conditional probability) để hiểu **quy tắc Bayes** (Bayes' rule).
> Ví dụ về xác suất có điều kiện: Đoán thời tiết dễ hơn nếu biết điều kiện (ví dụ: ở California vào mùa đông).

![03_Bayes'_Rule_Example](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/03_Bayes'_Rule_Example.png)

> **CONDITIONNAL PROBABILITIES** giúp chúng ta **GIẢM SAMPLE SEARCH SPACE**
- Để rút ra quy tắc Bayes, chúng ta bắt đầu bằng cách xem xét các tweet chỉ chứa từ "happy" (hạnh phúc) (vòng tròn màu xanh lam), thay vì toàn bộ tài liệu.
- **Xác suất một tweet là tích cực, *do* nó chứa từ "happy"** $P(\text{tích cực} | \text{"happy"})$ = (Số tweet tích cực VÀ chứa "happy") / (Tổng số tweet chứa "happy").
- Ví dụ: 75% khả năng tweet là tích cực nếu nó chứa từ "happy".
- Tương tự (vùng màu tím), **xác suất một tweet tích cực chứa từ "happy"** $P(\text{"happy"} | \text{tích cực})$ = (Số tweet tích cực VÀ chứa "happy") / (Tổng số tweet tích cực).
 
- Ví dụ: 3 trên 13 (0.231).
- **Xác suất có điều kiện** $P(B|A)$ là xác suất của kết quả B, biết rằng sự kiện A đã xảy ra.

![04_Bayes'_Rule_Equations](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W2/04_Bayes'_Rule_Equations.png)

- Sử dụng biểu đồ Venn, phương trình là:
    $P(\text{tích cực} | \text{"happy"}) = P(\text{giao điểm tích cực và "happy"}) / P(\text{"happy"})$
- Bạn có thể viết một phương trình tương tự bằng cách hoán đổi các điều kiện:
    $P(\text{"happy"} | \text{tích cực}) = P(\text{giao điểm tích cực và "happy"}) / P(\text{tích cực})$
- **Giao điểm** (intersection) là như nhau trong cả hai phương trình. Bằng cách "thao tác đại số" (algebraic manipulation), bạn có thể kết hợp chúng.
- Đây là biểu hiện của **Quy tắc Bayes**.
- **Công thức chung:** $P(x | y) = P(y | x) \times \frac{P(x)}{P(y)}$ (xác suất của x cho y bằng xác suất của y cho x nhân tỷ lệ xác suất của x trên xác suất của y).
- **Điểm rút ra chính:** Quy tắc Bayes dựa trên công thức toán học của xác suất có điều kiện. Nó cho phép bạn tính $P(x | y)$ nếu bạn biết $P(y | x)$ và tỷ lệ $P(x) / P(y)$.
- **CONDITIONAL PROBABILITIES** "LÀ" **BAYES' RULE**
- Video tiếp theo sẽ áp dụng quy tắc Bayes cho một mô hình gọi là **Naive Bayes** (Bayes ngây thơ) để xây dựng **bộ phân loại phân tích tâm lý** (sentiment analysis classifier).

### Naive Bayes Introduction

- Tuần này, bạn sẽ học cách phân loại tweet bằng **Naive Bayes** (Bayes ngây thơ), thay vì dùng **hồi quy logistic** (logistic regression) như tuần trước.
- Naive Bayes là một cơ sở (baseline) "nhanh chóng và bẩn thỉu" (quick and dirty) cho các nhiệm vụ **phân loại văn bản** (text classification).
- Đây là một ví dụ về **học máy có giám sát** (supervised machine learning).
- Nó được gọi là **"ngây thơ" (naive)** vì đưa ra giả định rằng các **tính năng (features)** (từ) là **độc lập (independent)** với nhau, điều mà trong thực tế hiếm khi xảy ra nhưng vẫn hoạt động tốt.

---

> Quy trình của Naive Bayes

1.  **Chuẩn bị dữ liệu**:
    - Bắt đầu với hai **corpora** (kho dữ liệu): một cho tweet tích cực và một cho tweet tiêu cực.
    - Trích xuất **từ vựng** (vocabulary) (tất cả các từ khác nhau) và **số lượng (counts)** của chúng trong mỗi kho (tích cực và tiêu cực).
2.  **Tính Tổng số từ**:
    - Đếm tổng số từ trong kho tích cực (ví dụ: 13) và tổng số từ trong kho âm (ví dụ: 12).
    - Đây là bước mới quan trọng cho Naive Bayes, cho phép tính **xác suất có điều kiện** (conditional probability).
3.  **Tính Bảng Xác suất Có điều kiện**:
    - Tính $P(\text{từ} | \text{lớp})$ bằng cách: (tần suất của từ trong lớp) / (tổng số từ trong lớp đó).
    - Ví dụ: $P(\text{"I"} | \text{positive}) = 3/13 \approx 0.24$. $P(\text{"I"} | \text{negative}) = 3/12 = 0.25$.
    - Lưu các giá trị này vào một bảng xác suất có điều kiện.
    - Tổng tất cả các xác suất cho mỗi lớp sẽ bằng 1.
4.  **Phân tích Bảng**:
    - Các từ có xác suất gần giống nhau ở cả hai lớp (ví dụ: "I", "am", "learning", "NLP") là các **từ trung lập (neutral words)** và không đóng góp nhiều vào tình cảm.
    - Các từ có **sự khác biệt đáng kể** (ví dụ: "happy", "sad", "not") là **"từ quyền lực" (powerful words)** giúp xác định tình cảm.
5.  **Vấn đề và Giải pháp (Sơ lược)**:
    * Nếu một từ (ví dụ: "because") chỉ xuất hiện trong kho tích cực, xác suất của nó ở kho âm sẽ là 0.
    * Điều này gây ra vấn đề tính toán. Để tránh điều này, bạn cần **làm mịn (smoothing)**.

---

#### Áp dụng (Suy luận)

- Giả sử có tweet mới: “Hôm nay tôi hạnh phúc, tôi đang học hỏi.”
- Sử dụng **quy tắc điều kiện suy luận Naive Bayes (Naive Bayes inference conditional rule)** để phân loại nhị phân.
- Quy tắc này lấy **tích (product)** của tỷ lệ (xác suất dương / xác suất âm) của mỗi từ trong tweet:
    $\prod \frac{P(\text{word}_i | \text{positive})}{P(\text{word}_i | \text{negative})}$
- **Ví dụ tính toán**:
    + "I": 0.2 / 0.2
    + "am": 0.2 / 0.2
    + "happy": 0.14 / 0.10
    + "Today": Không có trong từ vựng, vì vậy **không bao gồm (not include)** trong điểm số.
    + "I" (lần 2): 0.2 / 0.2
    + "I'm" (lần 2): 0.2 / 0.2
    + "learning": 0.10 / 0.10
- Các từ trung lập (tỷ lệ = 1) sẽ tự **hủy bỏ (cancel out)**.
- Kết quả cuối cùng là $0.14 / 0.10 = 1.4$.
- Vì giá trị này **cao hơn một** (1.4 > 1), tweet được kết luận là tích cực.
- Video tiếp theo sẽ xem xét các vấn đề và **đơn giản hóa các tính toán (simplify calculations)**.



















### Laplacian Smoothing
### Log Likelihood
### Training Naive Bayes
### Testing Naive Bayes
### Applications og Naive Bayes
### Naive Bayes Assumption
### Error Analysis


