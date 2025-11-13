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

![02_Markov_Chains](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/02_Markov_Chains.png)

> Để mô hình hóa xác suất một cách đúng đắn, chúng ta cần xác định xác suất của các nhãn từ loại (POS) và của các từ.

![03_Markov_Chains](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/03_Markov_Chains.png)

> Các vòng tròn trong đồ thị đại diện cho các trạng thái của mô hình. Một trạng thái đề cập đến một điều kiện nhất định của thời điểm hiện tại. Bạn có thể nghĩ về chúng như là các nhãn từ loại của từ hiện tại.

> Q = {q1, q2, q3} là tập hợp tất cả các trạng thái trong mô hình của bạn.

---
### **Markov Chains and POS Tags**
---

- Phần này giới thiệu cách bạn đi từ trạng thái (state) này sang trạng thái khác, xác định một thuật ngữ gọi là **xác suất chuyển tiếp** (transition probabilities).


#### Chuỗi Markov và POS Tagging

* Bạn có thể biểu diễn một câu (chuỗi các từ) bằng một biểu đồ.
* **Trạng thái (States):** Chính là các **phần của thẻ giọng nói** (POS tags). (Ví dụ: `NN` cho danh từ, `VB` cho động từ, `Other` cho các thẻ khác).
* **Cạnh (Edges):** Các mũi tên có trọng số, đại diện cho **xác suất chuyển tiếp**. Chúng xác định xác suất đi từ trạng thái này sang trạng thái khác.

#### Property Thuộc tính Markov (Markov Property)

* Đây là một đặc tính quan trọng giúp giữ cho mô hình đơn giản.
* **Định nghĩa:** Xác suất của sự kiện *tiếp theo* **chỉ phụ thuộc vào sự kiện *hiện tại***.
* Nó không cần thông tin từ bất kỳ trạng thái nào trước đó (lịch sử).
* **Ví dụ (POS Tagging):** Để biết xác suất từ tiếp theo (sau "learn") là danh từ, bạn chỉ cần biết trạng thái *hiện tại* là động từ (`VB`). Xác suất này chính là xác suất chuyển tiếp từ `VB` đến `NN` (ví dụ: 0.4).


#### Ma trận Chuyển tiếp (Transition Matrix)

* Bạn có thể sử dụng một **bảng** (table) để lưu trữ các xác suất chuyển tiếp. Đây là một biểu diễn tương đương và nhỏ gọn hơn của chuỗi Markov.
* Bảng này được gọi là **ma trận chuyển tiếp (A)**.
* **Kích thước:** N x N (với N là số trạng thái).
* **Hàng (Rows):** Đại diện cho **trạng thái hiện tại** (ví dụ: `NN`).
* **Cột (Columns):** Đại diện cho **trạng thái tương lai** (ví dụ: `NN`, `VB`, `Other`).
* **Giá trị:** Xác suất chuyển đổi (ví dụ: $P(\text{NN} | \text{NN})$, $P(\text{VB} | \text{NN})$).
* **Quy tắc quan trọng:** Tổng tất cả các xác suất chuyển tiếp đi từ một trạng thái nhất định (tức là, **tổng của mỗi hàng**) phải **luôn luôn bằng 1**.

#### Trạng thái Ban đầu (Initial State)

* **Vấn đề:** Mô hình không cho bạn biết cách gán thẻ POS cho **từ đầu tiên** trong câu (vì không có từ *trước đó*).
* **Giải pháp:** Giới thiệu một **trạng thái ban đầu** (initial state) (thường ký hiệu là $\pi$).
* Trạng thái này được thêm vào ma trận A, làm cho nó có kích thước **(N+1) x N**. Hàng đầu tiên (hàng 0) chứa các xác suất ban đầu (ví dụ: xác suất câu bắt đầu bằng `NN`, `VB`, v.v.).

* **Video tiếp theo:** Sẽ tìm hiểu về **Mô hình Markov ẩn** (Hidden Markov Models).

> Để giúp xác định các loại từ cho mỗi từ, bạn cần xây dựng một ma trận chuyển tiếp cung cấp cho bạn xác suất từ trạng thái này sang trạng thái khác.

![04_Markov_Chainsand_POS_Tags](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/04_Markov_Chainsand_POS_Tags.png)

> Trong sơ đồ ở trên, các vòng tròn màu xanh tương ứng với các nhãn loại từ, và các mũi tên tương ứng với xác suất chuyển tiếp từ loại từ này sang loại từ khác. Bạn có thể điền bảng bên phải từ sơ đồ bên trái. Hàng đầu tiên trong ma trận **A** của bạn tương ứng với phân phối ban đầu giữa tất cả các trạng thái. Theo bảng, câu có 40% khả năng bắt đầu bằng danh từ, 10% khả năng bắt đầu bằng động từ, và 50% khả năng bắt đầu bằng một nhãn loại từ khác.

> Trong ký hiệu tổng quát hơn, bạn có thể viết ma trận chuyển tiếp **A**, cho trước một số trạng thái **Q**, như sau:

![05_Markov_Chainsand_POS_Tags](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/05_Markov_Chainsand_POS_Tags.png)

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
