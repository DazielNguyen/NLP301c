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

#### Mô hình Markov ẩn (HMM)

- Video này giới thiệu về **Mô hình Markov ẩn (Hidden Markov Models - HMMs)**.
- Chúng được sử dụng để giải mã (decode) các **trạng thái ẩn (hidden states)** của một từ. Trong trường hợp của chúng ta, các trạng thái ẩn chính là **phần của lời nói (parts of speech - POS)** của từ đó.
- Video này cũng giới thiệu một khái niệm mới: **xác suất phát xạ (emission probabilities)**.

#### Ẩn (Hidden) vs. Quan sát được (Observable)

* Tên "ẩn" ngụ ý rằng các trạng thái (thẻ POS như danh từ, động từ) **không thể quan sát trực tiếp** (not directly observable) từ góc độ của máy móc.
* Mặc dù con người có thể nhìn vào từ "nhảy" (jump) và biết đó là động từ, nhưng máy tính chỉ nhìn thấy văn bản "nhảy" và không tự động biết thẻ của nó.
* **Thể quan sát được (Observables):** Đây là các **từ thực tế** (actual words) mà máy tính có thể thấy (ví dụ: "nhảy", "chạy", "bay").

#### Các thành phần của HMM

Mô hình HMM có hai bộ xác suất chính:

**1. Xác suất chuyển tiếp (Transition Probabilities)**

* Giống như Chuỗi Markov (đã học trước đó), HMM có **xác suất chuyển tiếp**, được biểu diễn bằng **ma trận A** (kích thước N x N, với N là số trạng thái).
* Chúng mô tả xác suất chuyển từ trạng thái ẩn này (ví dụ: `động từ`) sang trạng thái ẩn khác (ví dụ: `danh từ`).

**2. Xác suất phát xạ (Emission Probabilities)**
* Đây là các xác suất bổ sung. Chúng mô tả sự chuyển đổi từ một **trạng thái ẩn** (ví dụ: vòng tròn `động từ`) sang một **thể quan sát được** (ví dụ: hình chữ nhật `ăn`).
* **Ma trận phát xạ (B):**
    * Đây là một bảng (kích thước N x V, với V là số từ trong kho) nơi:
    * **Hàng** (Rows) = Trạng thái ẩn (N trạng thái).
    * **Cột** (Columns) = Thể quan sát được (V từ).
    * **Ví dụ:** $P(\text{từ 'ăn'} | \text{trạng thái 'động từ'}) = 0.5$. Điều này có nghĩa là khi mô hình ở trạng thái "động từ", có 50% khả năng nó sẽ "phát ra" (emit) từ "ăn".
* **Quy tắc:** Giống như ma trận A, **tổng của mỗi hàng** (sum of each row) trong ma trận phát xạ B phải bằng 1.

#### Tại sao cần Xác suất Phát xạ?

* Bởi vì các từ có thể có các thẻ POS khác nhau tùy thuộc vào **ngữ cảnh** (context).
* **Ví dụ:** Từ "trở lại" (back) có thể là một **danh từ** (trong "anh nằm ngửa" - he lay on his back) hoặc một **trạng từ** (trong "tôi sẽ trở lại" - I'll be back). Xác suất phát xạ giúp mô hình hóa khả năng này.

> Trong video trước, tôi đã chỉ cho bạn một ví dụ với mô hình Markov đơn giản. **Xác suất chuyển tiếp (transition probabilities)** cho phép bạn xác định xác suất chuyển tiếp từ một loại từ (POS) này sang loại từ khác. Bây giờ chúng ta sẽ khám phá các mô hình Markov ẩn. Trong các mô hình Markov ẩn, bạn sử dụng **xác suất phát ra (emission probabilities)**, xác suất này cho bạn biết khả năng chuyển từ một trạng thái (nhãn POS) sang một từ cụ thể.

![06_Hidden_Markov_Models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/06_Hidden_Markov_Models.png)

> Ví dụ, giả sử bạn đang ở trạng thái động từ, bạn có thể đi đến các từ khác với một số xác suất nhất định. **Ma trận phát xạ B** này sẽ được sử dụng cùng với **ma trận chuyển tiếp A** của bạn, để giúp bạn xác định từ loại của một từ trong câu. Để điền vào **ma trận B** của bạn, bạn chỉ cần có một tập dữ liệu đã được gán nhãn và tính xác suất từ một từ loại chuyển sang mỗi từ trong từ vựng của bạn. Dưới đây là tóm tắt những gì bạn đã học được cho đến nay:

![07_Hidden_Markov_Models](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/07_Hidden_Markov_Models.png)

> Lưu ý rằng tổng của mỗi hàng trong **ma trận A và B** của bạn phải bằng 1. Tiếp theo, tôi sẽ chỉ cho bạn cách bạn có thể tính các xác suất bên trong các ma trận này.

---
### **Calculating Probabilities**
---

- Phần này giải thích cách tính toán xác suất cho **ma trận chuyển tiếp (transition matrix)** (Ma trận A) từ một **kho tài liệu (corpus)** đã được gán thẻ.

#### 1. Ý tưởng Khái niệm (Conceptual Idea)

* **Mục tiêu:** Tính xác suất chuyển tiếp, ví dụ: $P(\text{Thẻ Tím} | \text{Thẻ Xanh})$.
* **Cách làm:**
    1.  **Đếm (Count):** Đếm số lần tổ hợp (cặp thẻ) đó xuất hiện. (Ví dụ: "Xanh" theo sau là "Tím" xuất hiện **2 lần**).
    2.  **Đếm Tổng:** Đếm tổng số lần "Thẻ Xanh" xuất hiện (tức là, tất cả các cặp bắt đầu bằng "Xanh"). (Ví dụ: **3 lần**).
    3.  **Tính toán:** Xác suất = (Số lần đếm) / (Tổng số lần đếm) = **2/3**.

#### 2. Công thức Chính thức (Formal Formula)

Để tính toán ma trận chuyển tiếp (xác suất $P(t_i | t_{i-1})$), bạn cần hai phép đếm:

1.  **$C(t_{i-1}, t_i)$:** (Tử số) Số lần thẻ $t_{i-1}$ được theo sau ngay lập tức bởi thẻ $t_i$ trong kho tài liệu.
2.  **$C(t_{i-1})$:** (Mẫu số) Tổng số lần thẻ $t_{i-1}$ xuất hiện (tức là, tổng của $C(t_{i-1}, t_j)$ cho mọi $t_j$ có thể).

Công thức là:
$$P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$$



#### 3. Chuẩn bị Kho tài liệu (Corpus Preparation)

Để tính toán các xác suất này một cách chính xác (sử dụng ví dụ về bài thơ Haiku), bạn cần chuẩn bị kho tài liệu:

* Coi mỗi dòng (line) là một **câu riêng biệt**.
* Thêm một **mã thông báo bắt đầu** (start token) vào mỗi câu. (Điều này rất quan trọng để tính toán **xác suất ban đầu** - initial probabilities, tức là hàng đầu tiên của ma trận A).
* Chuyển đổi tất cả các từ thành **chữ thường** (lowercase) để mô hình không phân biệt chữ hoa/thường.
* **Dấu câu** (Punctuation) được giữ nguyên (trong mô hình đồ chơi này).

* **Kết luận:** Bạn đã học cách lấy **số đếm (counts)** và biến chúng thành **xác suất (probabilities)**.
> Đây là một biểu đồ trực quan về cách tính xác suất:

![08_Calculating_Probabilities](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/08_Calculating_Probabilities.png)

> Số lần màu xanh lam (blue) được theo sau bởi màu tím (purple) là 2 trên 3. Chúng ta sẽ sử dụng logic tương tự để điền vào **ma trận chuyển tiếp** (transition matrices) và **ma trận phát xạ** (emission matrices) của mình. Trong ma trận chuyển tiếp, chúng ta sẽ đếm số lần cặp thẻ $t_{i-1}, t_i$ xuất hiện gần nhau và chia cho tổng số lần $t_{i-1}$ xuất hiện (điều này tương tự như số lần nó xuất hiện và được theo sau bởi bất cứ thứ gì khác).

![09_Calculating_Probabilities](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/09_Calculating_Probabilities.png)

> $C(t_{i-1}, t_i)$ là số lần mà nhãn (i-1) xuất hiện trước nhãn i. Từ đó, bạn có thể tính xác suất một nhãn xuất hiện sau nhãn khác.

---
### **Populating the Transition Matrix**
---

* Phần này hướng dẫn cách điền (populate) **ma trận chuyển tiếp** (transition matrix) (Ma trận A) bằng cách tính toán xác suất.
* Bạn cũng sẽ học về **làm mịn** (smoothing) để xử lý các vấn đề dữ liệu.

#### 1. Tính toán Số đếm (Counts)

* **Ma trận chuyển tiếp** có các **hàng** (rows) đại diện cho **trạng thái hiện tại** (current state) (thẻ POS $t_{i-1}$) và các **cột** (columns) đại diện cho **trạng thái tiếp theo** (next state) (thẻ POS $t_i$).
* Bạn bắt đầu bằng cách điền **số lượng (counts)** của tất cả các tổ hợp thẻ (cặp thẻ) vào ma trận.
* **Ví dụ (từ kho tài liệu):**
    * (Hàng `START`, Cột `NN`): Một danh từ (`NN`) sau mã thông báo bắt đầu $\rightarrow$ **1 lần**.
    * (Hàng `NN`, Cột `NN`): Một danh từ sau một danh từ $\rightarrow$ **0 lần**.
    * (Hàng `O`, Cột `NN`): Một danh từ sau một thẻ khác (`O`) $\rightarrow$ **6 lần**.
    * (Hàng `START`, Cột `O`): Thẻ `O` sau mã thông báo bắt đầu $\rightarrow$ **2 lần**.
    * (Hàng `NN`, Cột `O`): Thẻ `O` sau thẻ `NN` $\rightarrow$ **6 lần**.
    * (Hàng `O`, Cột `O`): Thẻ `O` sau thẻ `O` $\rightarrow$ **8 lần**.
* (Script lưu ý rằng trong ví dụ này, không có thẻ `VB` (động từ), vì vậy tất cả các số đếm liên quan đến `VB` đều bằng 0).

#### 2. Tính toán Xác suất (Chuẩn hóa)

* Sau khi có ma trận số đếm (tử số), bạn tính toán **xác suất chuyển tiếp**.
* Bạn **chia mỗi số đếm** trong một hàng cho **tổng của hàng đó** (row sum).
* (Tổng hàng đại diện cho tổng số lần trạng thái hiện tại (hàng đó) xuất hiện).
* **Ví dụ:** $P(\text{Other} | \text{NN})$ = (Số đếm `NN` $\rightarrow$ `O`) / (Tổng hàng `NN`) = 6 / 14.

#### 3. Vấn đề (Số 0)

Việc chuẩn hóa trực tiếp này gây ra hai vấn đề:

1.  **Xác suất bằng 0:** Rất nhiều mục nhập trong ma trận là 0. Điều này có nghĩa là mô hình sẽ không **khái quát hóa** (generalize) được cho các dữ liệu mới (ví dụ: một câu haiku có động từ).
2.  **Chia cho 0:** Nếu một hàng không có dữ liệu (ví dụ: tổng hàng `VB` là 0), bạn sẽ gặp lỗi chia cho 0.

#### 4. Giải pháp: Làm mịn (Smoothing)

* Để xử lý điều này, bạn thay đổi công thức bằng cách sử dụng **làm mịn** (tương tự như làm mịn Laplace/cộng thêm).
* **Công thức mới:**
    * **Tử số (Numerator):** Thêm một giá trị nhỏ **Epsilon ($\epsilon$)** vào *mỗi số đếm*.
        * $\text{Count}(t_{i-1}, t_i) + \epsilon$
    * **Mẫu số (Denominator):** Thêm **N * Epsilon ($N \times \epsilon$)** vào *tổng hàng* (với N là tổng số thẻ/trạng thái).
        * $\text{Count}(t_{i-1}) + (N \times \epsilon)$
* **Kết quả:**
    * Không còn bất kỳ mục nhập giá trị 0 nào.
    * Các trạng thái không có dữ liệu (như `VB`) giờ đây có xác suất phân bổ đều (khả năng tương đương nhau), điều này hợp lý vì không có dữ liệu để ước tính chúng.
* **Lưu ý:** Trong thực tế, bạn có thể **không muốn áp dụng làm mịn** cho **hàng đầu tiên** (xác suất ban đầu), để ngăn mô hình cho phép một câu bắt đầu bằng các thẻ không mong muốn (như dấu câu).

> Để điền đầy ma trận chuyển tiếp, bạn phải theo dõi số lần mỗi thẻ xuất hiện trước một thẻ khác.

![10_Populating_the_Transition_Matrix](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/10_Populating_the_Transition_Matrix.png)

> Trong bảng trên, bạn có thể thấy màu xanh lá cây tương ứng với danh từ (NN), màu tím tương ứng với động từ (VB), và màu xanh dương tương ứng với các loại khác (O). Màu cam ($π$) tương ứng với trạng thái ban đầu. Các con số bên trong ma trận tương ứng với số lần một nhãn từ loại xuất hiện ngay sau nhãn từ loại khác.
> Để đi từ O đến NN hay nói cách khác là để tính $P(N N O)$, bạn phải tính toán các giá trị sau:

![11_Populating_the_Transition_Matrix](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/11_Populating_the_Transition_Matrix.png)

> Tóm lại:

$$P\left(t_i \mid t_{i-1}\right) = \frac{C\left(t_{i-1}, t_i\right)}{\sum_{j=1}^{N} C\left(t_{i-1}, t_j\right)}$$

> Thật không may, đôi khi bạn có thể không thấy hai nhãn từ loại đứng liền kề nhau. Điều này sẽ cho bạn xác suất bằng 0. Để giải quyết vấn đề này, bạn sẽ "làm mịn" nó như sau:

![12_Populating_the_Transition_Matrix](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W2/12_Populating_the_Transition_Matrix.png)


> Thuộc tính $ε$ epsilon cho phép bạn không có bất kỳ hai chuỗi nào hiển thị với xác suất bằng 0. Tại sao điều này lại quan trọng?
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
