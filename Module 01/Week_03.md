# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 3: Vector Space Models
### Vector Space Models

* Tuần này, sẽ học về **không gian vectơ (vector spaces)** và loại thông tin mà chúng có thể mã hóa.
* Giới thiệu ý tưởng chung đằng sau **mô hình không gian vector (vector space models)**, lợi thế và ứng dụng của chúng trong **NLP** (Xử lý ngôn ngữ tự nhiên).

---

#### Khả năng của Mô hình Không gian Vectơ

1.  **Nắm bắt ý nghĩa (Semantic Meaning):**
    * Các mô hình không gian vectơ giúp xác định các câu có ý nghĩa giống nhau, ngay cả khi chúng *không* chia sẻ cùng một từ (ví dụ: hai câu hỏi khác nhau nhưng có cùng ý nghĩa).
    * Chúng cũng giúp phân biệt các câu có từ giống hệt nhau nhưng ý nghĩa khác nhau (ví dụ: "Bạn đang đi đâu?" so với "Bạn đến từ đâu?").
    * Ứng dụng: Xác định sự tương đồng cho câu trả lời, diễn giải và tóm tắt câu hỏi.

2.  **Nắm bắt sự phụ thuộc giữa các từ (Dependencies):**
    * Chúng có thể nắm bắt các mối quan hệ và sự phụ thuộc giữa các từ.
    * Ví dụ 1: Các từ "ngũ cốc" (cereal) và "bát" (bowl) có liên quan.
    * Ví dụ 2: Trong câu "bạn mua một cái gì đó và người khác bán nó", nửa sau của câu phụ thuộc vào nửa đầu.

---

#### Ứng dụng

* Các mô hình không gian vectơ được sử dụng trong:
    * **Trích xuất thông tin (Information extraction)** (trả lời câu hỏi ai, cái gì, ở đâu, như thế nào).
    * **Dịch máy (Machine translation)**.
    * Lập trình thể thao cờ vua (Chess sport programming) và nhiều ứng dụng khác.

---

#### Khái niệm cốt lõi

* Trích dẫn của **John Firth**, một nhà ngôn ngữ học Anh: "**Bạn sẽ biết một từ bởi công ty mà nó giữ**" (You shall know a word by the company it keeps).
* Đây là một trong những khái niệm cơ bản nhất trong NLP.
* Mô hình không gian vectơ thực hiện điều này bằng cách **xác định ngữ cảnh (context)** xung quanh mỗi từ, từ đó nắm bắt được **ý nghĩa tương đối (relative meaning)**.
* **Kết luận (Eureka):** Mô hình không gian vectơ cho phép bạn biểu diễn các từ và tài liệu dưới dạng **vectơ (vectors)**, nắm bắt được ý nghĩa tương đối.
* **Video tiếp theo:** Bạn sẽ xây dựng chúng từ đầu bằng cách sử dụng **ma trận đồng xuất hiện (co-occurrence matrices)**.
### Word by Word and Word by Doc
### Euclidean Distance
### Cosine Similarity: Intuition
### Cosine Similarity
### Manipulating Word in Vectors Spaces
### Visualization and PCA
### PCA Algorithmn
### The Rotation Matrix (Optional Reading)