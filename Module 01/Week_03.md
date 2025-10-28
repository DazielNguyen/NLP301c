# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 3: Vector Space Models

---
### Vector Space Models
---

- Tuần này, sẽ học về **không gian vectơ (vector spaces)** và loại thông tin mà chúng có thể mã hóa.
- Giới thiệu ý tưởng chung đằng sau **mô hình không gian vector (vector space models)**, lợi thế và ứng dụng của chúng trong **NLP** (Xử lý ngôn ngữ tự nhiên).

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

#### Ứng dụng

- Các mô hình không gian vectơ được sử dụng trong:
    + **Trích xuất thông tin (Information extraction)** (trả lời câu hỏi ai, cái gì, ở đâu, như thế nào).
    + **Dịch máy (Machine translation)**.
    + Lập trình **chatbots** và nhiều ứng dụng khác.

#### Khái niệm cốt lõi

- Trích dẫn của **John Firth**, một nhà ngôn ngữ học Anh: "**Bạn sẽ biết một từ bởi công ty mà nó giữ**" (You shall know a word by the company it keeps).
- Đây là một trong những khái niệm cơ bản nhất trong NLP.
- Mô hình không gian vectơ thực hiện điều này bằng cách **xác định ngữ cảnh (context)** xung quanh mỗi từ, từ đó nắm bắt được **ý nghĩa tương đối (relative meaning)**.
- **Kết luận (Eureka):** Mô hình không gian vectơ cho phép bạn biểu diễn các từ và tài liệu dưới dạng **vectơ**, nắm bắt được ý nghĩa tương đối.

---
### Word by Word and Word by Doc
---

- Hướng dẫn cách xây dựng **vectơ** (vectors) dựa trên **ma trận đồng xuất hiện** (co-occurrence matrices).
* Tùy thuộc vào nhiệm vụ, bạn có thể có một số **thiết kế (designs)** khả thi để mã hóa một từ hoặc tài liệu thành vectơ.

#### Hai thiết kế Mô hình Không gian Vectơ
> Word by Word Design

![02_Word_by_Word](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/02_Word_by_Word.png)

1.  **Thiết kế Từng từ (Word-by-word)**
    - Bạn tạo một ma trận đồng xuất hiện và trích xuất vectơ (bản trình bày) cho các từ trong kho (corpus) của bạn.
    - **Sự đồng xuất hiện (Co-occurrence)** của hai từ là số lần chúng xuất hiện cùng nhau trong một **khoảng cách từ nhất định k** (a certain word distance k).
    - **Ví dụ:** Với $k=2$, nếu "dữ liệu" (data) và "đơn giản" (simple) cùng xuất hiện 2 lần (một lần cách 1 từ, một lần cách 2 từ), giá trị trong ma trận là 2.
    - **Hàng (row)** của ma trận đồng xuất hiện (ví dụ: hàng cho từ "dữ liệu") trở thành **biểu diễn vectơ** của từ đó (ví dụ: [2, 1, 1, 0]).

> Word by Document Design

![03_Word_by_Document](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/03_Word_by_Document.png)

2.  **Thiết kế Từ theo tài liệu (Word-by-document)**
    - Quá trình này khá giống nhau, nhưng bạn đếm số lần các từ xuất hiện trong các **tài liệu (documents)** thuộc các **danh mục (categories)** cụ thể.
    - **Ví dụ:** Một kho tài liệu có 3 danh mục: giải trí, kinh tế, và học máy.
    - Từ "dữ liệu" (data) xuất hiện: 500 lần (giải trí), 6.620 lần (kinh tế), 9.320 lần (học máy).
    - Từ "phim" (movie) xuất hiện: 7.000 lần (giải trí), 4.000 lần (kinh tế), 1.000 lần (học máy).

#### Xây dựng Không gian Vectơ và Sự tương đồng
> Không gian Vectơ và Sự tương đồng

![04_Vector_Space_and_Similarity](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/04_Vector_Space_and_Similarity.png)

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

---
### Euclidean Distance
---

- Giới thiệu về **khoảng cách Euclide** (Euclidean distance), một **số liệu tương đồng** (similarity metric) dùng để xác định hai điểm (hoặc vectơ) cách nhau bao xa.

#### **Trường hợp 2 chiều (2D):**

> Tính khoảng cách Euclidian giữa 2 vector

![05_Euclidian_Distance](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/05_Euclidian_Distance.png)

- Sử dụng ví dụ về hai vectơ corpora ("giải trí" và "học máy") với hai chiều là số lần xuất hiện của từ "dữ liệu" và "phim".
- Khoảng cách Euclide là **chiều dài của đoạn đường thẳng** (length of the line segment) nối hai vectơ đó trong không gian.
- Công thức được sử dụng là một ví dụ của **định lý Pythagore** (Pythagorean theorem): 

- $d(B,A)=\sqrt{(\text{khoảng cách ngang})^2 + (\text{khoảng cách dọc})^2}$.
- Trong ví dụ, kết quả xấp xỉ 10.667.

#### **Trường hợp Kích thước cao hơn (n-Dimension):**

> Tính tổng quát hóa việc tìm khoảng cách giữa hai điểm (A, B) sang khoảng cách giữa một vectơ n chiều như sau:

![06_Euclidian_Distance_Generalization](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/06_Euclidian_Distance_Generalization.png)

> Ví dụ tính khoảng cách giữa 2 vector (n = 3).

![07_Example_Euclidian_Distance_Generalization](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/07_Example_Euclidian_Distance_Generalization.png)

- Quy trình này là sự **khái quát hóa** (generalization) của trường hợp 2D.
- Để tìm khoảng cách giữa hai vectơ (ví dụ: 'ice-cream' và 'boba'):
    1.  Lấy **sự khác biệt** (difference) giữa mỗi kích thước.
    2.  **Bình phương** (Square) những khác biệt đó.
    3.  **Tổng hợp** (Sum) chúng lại.
    4.  Lấy **căn bậc hai** (square root) của kết quả.
- Công thức này (từ đại số) được gọi là **định mức của sự khác biệt** (norm of the difference) giữa các vectơ.

#### **Triển khai trong Python:**

- Bạn có thể sử dụng mô-đun `linalg` (linear algebra) từ **NumPy**.
- Hàm `norm` (`np.linalg.norm(v - w)`) có thể tính toán định mức của chênh lệch (tức là khoảng cách Euclide) cho không gian n chiều.

#### **Điểm rút ra chính:**
- Khoảng cách Euclide về cơ bản là chiều dài của đường thẳng nối hai vectơ.
- Bằng cách sử dụng số liệu này, bạn có thể hiểu được hai tài liệu hoặc từ **giống nhau** (similar) như thế nào (khoảng cách càng nhỏ, càng giống nhau).

---
### Cosine Similarity: Intuition
---

- Phần này giới thiệu về **sự tương đồng cosin** (cosine similarity), một loại **chức năng tương đồng** (similarity function) khác.
- Về cơ bản, nó sử dụng **cosin của góc** (cosine of the angle) giữa hai vectơ để cho biết chúng có gần nhau hay không.
- Nó sẽ chỉ ra vấn đề của việc sử dụng **khoảng cách Euclide** (Euclidean distance) khi so sánh các tài liệu (corpora) và cách sự tương đồng cosin khắc phục điều này.

> Ví dụ Consine Similarity: Intuition

![08_Example_Consine_Similarity_Intuition](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/08_Example_Consine_Similarity_Intuition.png)

#### **Ví dụ về vấn đề của Khoảng cách Euclide:**

- Giả sử có một không gian vectơ với các từ "bệnh" (disease) và "trứng" (eggs).
- Có ba kho: **thực phẩm** (food), **nông nghiệp** (agriculture), và **lịch sử** (history).
- Kho "thực phẩm" có số lượng từ nhỏ, trong khi "nông nghiệp" và "lịch sử" có số lượng từ tương tự (lớn hơn).
- Khoảng cách Euclide $d_2$ (giữa nông nghiệp và lịch sử) **nhỏ hơn** khoảng cách $d_1$ (giữa thực phẩm và nông nghiệp).
- Điều này (sai lầm) cho thấy kho "nông nghiệp" và "lịch sử" giống nhau hơn.

#### **Giải pháp (Sự tương đồng Cosin):**

- Một phương pháp khác là tính cosin của góc trong (inner angle).
- Nếu góc nhỏ, cosin gần bằng **một (1)**. Nếu góc gần 90 độ, cosin gần **không (0)**.
- Trong ví dụ, **góc Alpha** (giữa thực phẩm và nông nghiệp) **nhỏ hơn** **góc Beta** (giữa nông nghiệp và lịch sử).
- Do đó, cosin của các góc là một **đại diện tốt hơn** (better representation) về sự tương đồng so với khoảng cách Euclide.

#### **Ưu điểm chính:** 
- Ưu điểm của số liệu này so với khoảng cách Euclide là nó **không bị sai lệch bởi sự khác biệt kích thước** (not biased by the size difference) giữa các biểu diễn.
- **Tóm lại:** Khoảng cách Euclide không lý tưởng cho các tài liệu có kích thước khác nhau. Sự tương đồng cosin sử dụng góc và do đó **không phụ thuộc vào kích thước** (independent of the size) của các corpus.

---
### Cosine Similarity
---

- Phần này hướng dẫn cách tính **tích dấu chấm** (dot product) và **định mức** (norm) của vectơ. Khi biết hai điều này, bạn sẽ có thể tính được **điểm tương đồng cosin** (cosine similarity score).
- Bạn sẽ học cách tính cosin của **góc trong** (inner angle) của hai vectơ và hiểu giá trị tương đồng cosin liên quan như thế nào đến sự **giống nhau của các hướng** (similarity of the directions).
- Cần nhớ lại các định nghĩa từ đại số:
    1.  **Định mức (Norm)** (hay **độ lớn - magnitude**) của một vectơ: Được định nghĩa là **căn bậc hai của tổng các phần tử bình phương** (square root of the sum of its squared elements) của nó.
    > Norm Equation
    $$\|\vec{v}\| = \sqrt{\sum_{i=1}^{n} v_{i}^{2}}$$

    2.  **Tích điểm (Dot product)** giữa hai vectơ: Là **tổng tích giữa các phần tử của chúng** (sum of the products between their elements) trong mỗi chiều
    + Ví dụ: Sử dụng hai corpora (thể tích nông nghiệp 'v' và kho lịch sử 'w') với các chiều là "bệnh" và "trứng". Góc giữa chúng là Beta.
    Đây là mã LaTeX cho công thức trong ảnh của bạn:
    > Dot Product Equation
    $$\vec{v} \cdot \vec{w} = \sum_{i=1}^{n} v_{i} \cdot w_{i}$$

#### **Công thức Tương đồng Cosin:** 
- Cosin của góc Beta bằng **tích chấm** giữa các vectơ chia cho **tích của hai định mức** (dot product between the vectors divided by the product of the two norms).
> Cosine Similarity Equation
$$\cos(\beta) = \frac{v \cdot w}{||v|| \times ||w||}$$

- Khi thay thế các giá trị (Tử số: tích điểm; Mẫu số: tích của các định mức), ví dụ cho kết quả tương đồng cosin là 0.87.

#### **Giải thích ý nghĩa số liệu:**
> Ví dụ về sự tương đồng và sự không tương đồng của 2 vector

![09_Example_Consine_Similarity](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W3/09_Example_Consine_Similarity.png)

- (Đối với không gian vectơ chỉ có giá trị dương bạn đã thấy cho đến nay):
- **Trực giao (Orthogonal)** (góc 90 độ): Cosin = 0. (Nghĩa là chúng **không giống nhau tối đa** - maximally dissimilar).
- **Cùng hướng (Same direction)** (góc 0 độ): Cosin = 1.

#### **Kết luận:** 
- Khi cosin của góc tiến gần đến 1, hướng của chúng càng gần.

#### **Điểm rút ra chính:**
- Số liệu này tỷ lệ thuận với sự giống nhau giữa các hướng của vectơ.
- Đối với các không gian vectơ dương đã thấy, sự tương đồng cosin có giá trị từ 0 đến 1.
#### **Tóm lại:**
- Tương đồng cosin của một vectơ với chính nó = 1.
- Nếu các vectơ vuông góc (perpendicular) = 0.
- Các vectơ tương tự có điểm (score) cao hơn.

---
### Manipulating Word in Vectors Spaces
---

---
### Visualization and PCA
---

---
### PCA Algorithmn
---

---
### The Rotation Matrix (Optional Reading)
---