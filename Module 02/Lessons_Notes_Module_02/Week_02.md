# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 2: Part of Speech Tagging and Hidden Markov Models**
---
### **Part of Speech Tagging**
---

#### Tổng quan về Tuần 2

- Bạn sẽ tìm hiểu các ứng dụng của nó và cách tính toán **độ chính xác** (accuracy) của trình gắn thẻ.
- Tuần này, bạn sẽ học về **Chuỗi Markov** (Markov Chains), **Mô hình Markov ẩn** (Hidden Markov Models - HMMs), và **thuật toán Viterbi** (Viterbi algorithm) để tạo các thẻ này.


#### Gắn thẻ Phần của Lời nói (POS Tagging) là gì?

- **Phần của lời nói** (Part of speech) đề cập đến loại từ hoặc thuật ngữ từ vựng trong một ngôn ngữ (ví dụ: **danh từ, động từ, tính từ, trạng từ, đại từ, giới từ**).
- Vì việc viết đầy đủ tên các thuật ngữ này (ví dụ: "Tại sao không học một cái gì đó?") rất **"cồng kềnh"** (cumbersome), bạn sẽ sử dụng một biểu diễn ngắn gọi là **thẻ** (tags).
- Quá trình gán các thẻ này cho các từ trong câu được gọi là **gắn thẻ phần của lời nói** (hoặc **POS tagging**).

#### Ứng dụng của POS Tagging

Thẻ POS mô tả cấu trúc và giúp đưa ra các giả định về ngữ nghĩa. Các ứng dụng bao gồm:

1.  **Xác định các thực thể được đặt tên** (Identifying named entities).
    * Ví dụ: Trong "Tháp Eiffel nằm ở Paris", "Tháp Eiffel" và "Paris" là các thực thể được đặt tên.
2.  **Độ phân giải đồng tham chiếu** (Co-reference resolution).
    * Ví dụ: Trong "Tháp Eiffel... **nó** cao 324 mét", POS tagging giúp suy ra "nó" (it) đề cập đến "Tháp Eiffel".
3.  **Nhận dạng giọng nói** (Speech recognition).
    * Được sử dụng để kiểm tra xem một chuỗi từ có xác suất cao hay không.
> Gán nhãn từ loại (POS) là quá trình gán một từ loại cho một từ. Khi làm vậy, bạn sẽ học được những điều sau:

- Markov Chains
- Hidden Markov Models
- Viterbi algorithm

> Dưới đây là một ví dụ cụ thể:

![01_Part_of_Speech_Tagging](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/01_Part_of_Speech_Tagging.png)

> Bạn có thể sử dụng gán nhãn từ loại cho:

- Xác định thực thể được đặt tên
- Nhận dạng giọng nói
- Giải quyết tham chiếu

> Bạn có thể sử dụng xác suất xuất hiện gần nhau của các nhãn POS để đưa ra kết quả hợp lý nhất.

---
### **Markov Chains**
---

- **Chuỗi Markov** (Markov chains) rất quan trọng vì chúng được sử dụng trong **nhận dạng giọng nói** (speech recognition) và **gắn thẻ phần của lời nói** (POS tagging).
- Phần này sẽ nói về **xác suất chuyển tiếp** (transition probabilities) và **trạng thái** (states).


#### Ứng dụng (Ví dụ về POS Tagging)

- **Câu hỏi:** Nếu bạn có một câu (ví dụ: "Why not learn"), và bạn biết "learn" là một **động từ** (verb), thì từ tiếp theo có khả năng là loại từ gì (danh từ, động từ, v.v.)?
- **Ý tưởng (Giả định Markov):** Khả năng (probability) của thẻ POS tiếp theo trong một câu có xu hướng **phụ thuộc vào thẻ POS của từ trước đó**.
- Ví dụ: Một động từ có nhiều khả năng được theo sau bởi một **danh từ** (noun) (ví dụ: xác suất 0.6) hơn là một động từ khác (ví dụ: xác suất 0.2).

#### Chuỗi Markov là gì?

- Chúng là một loại **mô hình ngẫu nhiên** (stochastic model) (ngẫu nhiên = random) mô tả một chuỗi các sự kiện.
- Để có được xác suất cho mỗi sự kiện, nó chỉ cần **trạng thái** (states) của các sự kiện *trước đó*.
- Một chuỗi Markov có thể được mô tả như một **biểu đồ có hướng** (directed graph).

#### Các thành phần của Chuỗi Markov

- **Trạng thái (States):**
    * Đây là các **vòng tròn** (circles) của biểu đồ.
    * Một "trạng thái" đề cập đến một "điều kiện nhất định của thời điểm hiện tại" (certain condition of the current moment).
    * *Ví dụ (Tương tự):* Nước có thể ở trạng thái đóng băng, lỏng, hoặc khí. Mỗi trạng thái này là một vòng tròn (q1, q2, q3).
    * Tập hợp tất cả các trạng thái được gọi là Q.
- **Xác suất chuyển tiếp (Transition Probabilities):**
    * Đây là các **mũi tên** (arrows) nối các trạng thái.
    * Các con số được liên kết với các mũi tên (ví dụ: 0.6, 0.2) cho biết khả năng di chuyển từ trạng thái này sang trạng thái khác.

#### Kết luận

- Trong bối cảnh của chúng ta, các **trạng thái** (states) này có thể được coi là các **phần của thẻ giọng nói** (POS tags) (ví dụ: một trạng thái cho "động từ", một trạng thái cho "danh từ", v.v.).
* **Video tiếp theo:** Sẽ tìm hiểu về các phần của thẻ giọng nói.

> Bạn có thể sử dụng chuỗi Markov để xác định xác suất của từ tiếp theo. Ví dụ dưới đây, bạn có thể thấy rằng từ có khả năng xuất hiện nhất sau một động từ là một danh từ.

![02_Part_of_Speech_Tagging](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/02_Part_of_Speech_Tagging.png)

> Để mô hình hóa xác suất một cách đúng đắn, chúng ta cần xác định xác suất của các nhãn từ loại (POS) và của các từ.

![03_Part_of_Speech_Tagging](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/03_Part_of_Speech_Tagging.png)

> Các vòng tròn trong đồ thị đại diện cho các trạng thái của mô hình. Một trạng thái đề cập đến một điều kiện nhất định của thời điểm hiện tại. Bạn có thể nghĩ về chúng như là các nhãn từ loại của từ hiện tại.

> Q = {q1, q2, q3} là tập hợp tất cả các trạng thái trong mô hình của bạn.

---
### **Markov Chains and POS Tags**
---



---
### **Hidden Markov Models**
---


---
### **Calculating Probabilities**
---



---
### **Populating the Transition Matrix**
---


---
### **Populating the Emission Matrix**
---


---
### **The Viterbi Algorithm**
---



---
### **Viterbi: Initialization**
---


---
### **Viterbi: Forward Pass**
---

---
### **Viterbi: Backward Pass**
---
