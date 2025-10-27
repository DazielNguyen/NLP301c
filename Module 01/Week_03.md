# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 3: Vector Space Models
### Vector Space Models

- Tuần này, sẽ học về **không gian vectơ (vector spaces)** và loại thông tin mà chúng có thể mã hóa.
- Giới thiệu ý tưởng chung đằng sau **mô hình không gian vector (vector space models)**, lợi thế và ứng dụng của chúng trong **NLP** (Xử lý ngôn ngữ tự nhiên).

---

#### Khả năng của Mô hình Không gian Vectơ

> Ví dụ

![01_Example_Application_Vector_Spaces_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/01_Example_Application_Vector_Spaces_Model.png)

1.  **Nắm bắt ý nghĩa (Semantic Meaning):**
    - Các mô hình không gian vectơ giúp xác định các câu có ý nghĩa giống nhau, ngay cả khi chúng *không* chia sẻ cùng một từ (ví dụ: hai câu hỏi khác nhau nhưng có cùng ý nghĩa).
    - Chúng cũng giúp phân biệt các câu có từ giống hệt nhau nhưng ý nghĩa khác nhau (ví dụ: "Bạn đang đi đâu?" so với "Bạn đến từ đâu?").
    - Ứng dụng: Xác định sự tương đồng cho câu trả lời, diễn giải và tóm tắt câu hỏi.

2.  **Nắm bắt sự phụ thuộc giữa các từ (Dependencies):**
    - Chúng có thể nắm bắt các mối quan hệ và sự phụ thuộc giữa các từ.
    - Ví dụ 1: Các từ "ngũ cốc" (cereal) và "bát" (bowl) có liên quan.
    - Ví dụ 2: Trong câu "bạn mua một cái gì đó và người khác bán nó", nửa sau của câu phụ thuộc vào nửa đầu.

---

#### Ứng dụng

- Các mô hình không gian vectơ được sử dụng trong:
    + **Trích xuất thông tin (Information extraction)** (trả lời câu hỏi ai, cái gì, ở đâu, như thế nào).
    + **Dịch máy (Machine translation)**.
    + Lập trình **chatbots** và nhiều ứng dụng khác.

---

#### Khái niệm cốt lõi

- Trích dẫn của **John Firth**, một nhà ngôn ngữ học Anh: "**Bạn sẽ biết một từ bởi công ty mà nó giữ**" (You shall know a word by the company it keeps).
- Đây là một trong những khái niệm cơ bản nhất trong NLP.
- Mô hình không gian vectơ thực hiện điều này bằng cách **xác định ngữ cảnh (context)** xung quanh mỗi từ, từ đó nắm bắt được **ý nghĩa tương đối (relative meaning)**.
- **Kết luận (Eureka):** Mô hình không gian vectơ cho phép bạn biểu diễn các từ và tài liệu dưới dạng **vectơ**, nắm bắt được ý nghĩa tương đối.

### Word by Word and Word by Doc

- Hướng dẫn cách xây dựng **vectơ** (vectors) dựa trên **ma trận đồng xuất hiện** (co-occurrence matrices).
* Tùy thuộc vào nhiệm vụ, bạn có thể có một số **thiết kế (designs)** khả thi để mã hóa một từ hoặc tài liệu thành vectơ.

---

#### Hai thiết kế Mô hình Không gian Vectơ
> Word by Word Design

![02_Word_by_Word]()

1.  **Thiết kế Từng từ (Word-by-word)**
    - Bạn tạo một ma trận đồng xuất hiện và trích xuất vectơ (bản trình bày) cho các từ trong kho (corpus) của bạn.
    - **Sự đồng xuất hiện (Co-occurrence)** của hai từ là số lần chúng xuất hiện cùng nhau trong một **khoảng cách từ nhất định k** (a certain word distance k).
    - **Ví dụ:** Với $k=2$, nếu "dữ liệu" (data) và "đơn giản" (simple) cùng xuất hiện 2 lần (một lần cách 1 từ, một lần cách 2 từ), giá trị trong ma trận là 2.
    - **Hàng (row)** của ma trận đồng xuất hiện (ví dụ: hàng cho từ "dữ liệu") trở thành **biểu diễn vectơ** của từ đó (ví dụ: [2, 1, 1, 0]).

> Word by Document Design

![03_Word_by_Document]()

2.  **Thiết kế Từ theo tài liệu (Word-by-document)**
    - Quá trình này khá giống nhau, nhưng bạn đếm số lần các từ xuất hiện trong các **tài liệu (documents)** thuộc các **danh mục (categories)** cụ thể.
    - **Ví dụ:** Một kho tài liệu có 3 danh mục: giải trí, kinh tế, và học máy.
    - Từ "dữ liệu" (data) xuất hiện: 500 lần (giải trí), 6.620 lần (kinh tế), 9.320 lần (học máy).
    - Từ "phim" (movie) xuất hiện: 7.000 lần (giải trí), 4.000 lần (kinh tế), 1.000 lần (học máy).

---

#### Xây dựng Không gian Vectơ và Sự tương đồng
> Không gian Vectơ và Sự tương đồng

![04_Vector_Space_and_Similarity]()

- Từ ma trận "Từ theo tài liệu", bạn có thể lấy biểu diễn cho *từ* (từ các hàng) hoặc cho *loại tài liệu* (từ các cột).
- **Ví dụ (lấy theo cột):** Không gian vectơ sẽ có hai chiều (tương ứng với từ "dữ liệu" và "phim").
    + Vectơ "giải trí" = [500, 7.000]
    + Vectơ "kinh tế" = [6.620, 4.000]
    + Vectơ "học máy" = [9.320, 1.000]
- Trong không gian này, có thể thấy rằng tài liệu "kinh tế" và "học máy" **giống nhau (similar)** hơn nhiều so với "giải trí".
- Sắp tới, bạn sẽ học cách so sánh các biểu diễn vectơ này bằng **sự tương đồng cosin (cosine similarity)** và **khoảng cách Euclide (Euclidean distance)**.

#### Kết luận

- Bạn đã thấy cách lấy không gian vectơ bằng hai thiết kế: **từng từ** và **từng văn bản (word-by-document)**.
- Bạn đã học cách xác định mối quan hệ (như **sự tương đồng - similarity**) giữa các loại tài liệu trong không gian vectơ.

### Euclidean Distance
### Cosine Similarity: Intuition
### Cosine Similarity
### Manipulating Word in Vectors Spaces
### Visualization and PCA
### PCA Algorithmn
### The Rotation Matrix (Optional Reading)