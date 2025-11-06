# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## **Week 3: Vector Space Models**

---
### **Vector Space Models**
---

- Tuần này, sẽ học về **không gian vectơ (vector spaces)** và loại thông tin mà chúng có thể mã hóa.
- Giới thiệu ý tưởng chung đằng sau **mô hình không gian vector (vector space models)**, lợi thế và ứng dụng của chúng trong **NLP** (Xử lý ngôn ngữ tự nhiên).

#### Khả năng của Mô hình Không gian Vectơ

> Ví dụ

![01_Example_Application_Vector_Spaces_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/01_Example_Application_Vector_Spaces_Model.png)

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
### **Word by Word and Word by Doc**
---

- Hướng dẫn cách xây dựng **vectơ** (vectors) dựa trên **ma trận đồng xuất hiện** (co-occurrence matrices).
* Tùy thuộc vào nhiệm vụ, bạn có thể có một số **thiết kế (designs)** khả thi để mã hóa một từ hoặc tài liệu thành vectơ.

#### Hai thiết kế Mô hình Không gian Vectơ
> Word by Word Design

![02_Word_by_Word](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/02_Word_by_Word.png)

1.  **Thiết kế Từng từ (Word-by-word)**
    - Bạn tạo một ma trận đồng xuất hiện và trích xuất vectơ (bản trình bày) cho các từ trong kho (corpus) của bạn.
    - **Sự đồng xuất hiện (Co-occurrence)** của hai từ là số lần chúng xuất hiện cùng nhau trong một **khoảng cách từ nhất định k** (a certain word distance k).
    - **Ví dụ:** Với $k=2$, nếu "dữ liệu" (data) và "đơn giản" (simple) cùng xuất hiện 2 lần (một lần cách 1 từ, một lần cách 2 từ), giá trị trong ma trận là 2.
    - **Hàng (row)** của ma trận đồng xuất hiện (ví dụ: hàng cho từ "dữ liệu") trở thành **biểu diễn vectơ** của từ đó (ví dụ: [2, 1, 1, 0]).

> Word by Document Design

![03_Word_by_Document](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/03_Word_by_Document.png)

2.  **Thiết kế Từ theo tài liệu (Word-by-document)**
    - Quá trình này khá giống nhau, nhưng bạn đếm số lần các từ xuất hiện trong các **tài liệu (documents)** thuộc các **danh mục (categories)** cụ thể.
    - **Ví dụ:** Một kho tài liệu có 3 danh mục: giải trí, kinh tế, và học máy.
    - Từ "dữ liệu" (data) xuất hiện: 500 lần (giải trí), 6.620 lần (kinh tế), 9.320 lần (học máy).
    - Từ "phim" (movie) xuất hiện: 7.000 lần (giải trí), 4.000 lần (kinh tế), 1.000 lần (học máy).

#### Xây dựng Không gian Vectơ và Sự tương đồng
> Không gian Vectơ và Sự tương đồng

![04_Vector_Space_and_Similarity](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/04_Vector_Space_and_Similarity.png)

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
### **Euclidean Distance**
---

- Giới thiệu về **khoảng cách Euclide** (Euclidean distance), một **số liệu tương đồng** (similarity metric) dùng để xác định hai điểm (hoặc vectơ) cách nhau bao xa.

#### **Trường hợp 2 chiều (2D):**

> Tính khoảng cách Euclidian giữa 2 vector

![05_Euclidian_Distance](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/05_Euclidian_Distance.png)

- Sử dụng ví dụ về hai vectơ corpora ("giải trí" và "học máy") với hai chiều là số lần xuất hiện của từ "dữ liệu" và "phim".
- Khoảng cách Euclide là **chiều dài của đoạn đường thẳng** (length of the line segment) nối hai vectơ đó trong không gian.
- Công thức được sử dụng là một ví dụ của **định lý Pythagore** (Pythagorean theorem): 

- $d(B,A)=\sqrt{(\text{khoảng cách ngang})^2 + (\text{khoảng cách dọc})^2}$.
- Trong ví dụ, kết quả xấp xỉ 10.667.

#### **Trường hợp Kích thước cao hơn (n-Dimension):**

> Tính tổng quát hóa việc tìm khoảng cách giữa hai điểm (A, B) sang khoảng cách giữa một vectơ n chiều như sau:

![06_Euclidian_Distance_Generalization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/06_Euclidian_Distance_Generalization.png)

> Ví dụ tính khoảng cách giữa 2 vector (n = 3).

![07_Example_Euclidian_Distance_Generalization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/07_Example_Euclidian_Distance_Generalization.png)

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
### **Cosine Similarity: Intuition**
---

- Phần này giới thiệu về **sự tương đồng cosin** (cosine similarity), một loại **chức năng tương đồng** (similarity function) khác.
- Về cơ bản, nó sử dụng **cosin của góc** (cosine of the angle) giữa hai vectơ để cho biết chúng có gần nhau hay không.
- Nó sẽ chỉ ra vấn đề của việc sử dụng **khoảng cách Euclide** (Euclidean distance) khi so sánh các tài liệu (corpora) và cách sự tương đồng cosin khắc phục điều này.

> Ví dụ Consine Similarity: Intuition

![08_Example_Consine_Similarity_Intuition](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/08_Example_Consine_Similarity_Intuition.png)

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
### **Cosine Similarity**
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

![09_Example_Consine_Similarity](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/09_Example_Consine_Similarity.png)

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
### **Manipulating Word in Vectors Spaces**
---

- Phần này hướng dẫn cách **thao tác vectơ** (vector manipulation) bằng cách sử dụng **số học vectơ đơn giản** (simple vector arithmetic), cụ thể là cộng và trừ vectơ.
- Mục tiêu là sử" dụng các phép toán này để **suy ra các mối quan hệ không xác định** (infer unknown relationships) giữa các từ, ví dụ như dự đoán thủ đô của một quốc gia.

#### Quy trình

> Ví dụ Tìm Thủ đô

![10_Example_Manipulating_Word_in_Vectors_Spaces](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/10_Example_Manipulating_Word_in_Vectors_Spaces.png)

1.  **Thiết lập:** Giả sử bạn có một không gian vectơ (ví dụ: 2D) chứa các quốc gia và thủ đô. Bạn biết thủ đô của **Hoa Kỳ (USA)** là **Washington DC** và muốn tìm thủ đô của **Nga (Russia)**.
> Không gian Vector 2D của ví dụ trên 

![11_Example_Manipulating_Word_in_Vectors_Spaces](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/11_Example_Manipulating_Word_in_Vectors_Spaces.png)

2.  **Tìm Vectơ Mối quan hệ:**
    - Đầu tiên, bạn tìm "vectơ mối quan hệ" (relationship vector) kết nối quốc gia với thủ đô của nó.
    - Cách làm: Lấy **sự khác biệt** (difference) giữa vectơ của thủ đô đã biết và quốc gia tương ứng (ví dụ: $V_{\text{Washington}} - V_{\text{USA}}$).
    - Vectơ kết quả này biểu thị "cần di chuyển bao nhiêu" từ một quốc gia để đến thủ đô của nó.
3.  **Áp dụng Vectơ Mối quan hệ (Dự đoán):**
    - Để tìm thủ đô của Nga, bạn **tính tổng** (sum) biểu diễn vectơ của "Nga" với vectơ mối quan hệ vừa tìm được ở bước trước (ví dụ: $V_{\text{Russia}} + (V_{\text{Washington}} - V_{\text{USA}})$).
4.  **Tìm kết quả gần nhất:**
    - Kết quả của phép cộng là một vectơ mới (ví dụ: [10, 4]).
    - Tuy nhiên, có thể không có thành phố nào chính xác tại vị trí đó.
    - Bạn phải tìm thành phố **giống nhất (most similar)** (gần nhất) với vectơ [10, 4] bằng cách so sánh nó với tất cả các vectơ thành phố khác, sử dụng **khoảng cách Euclide** (Euclidean distance) hoặc **điểm tương đồng cosin** (cosine similarity).
    - Trong ví dụ, vectơ gần nhất là "Moscow".

#### Kết luận

- **Mấu chốt:** Quá trình này chỉ hiệu quả khi bạn có một không gian vectơ nơi các biểu diễn (vectơ) **nắm bắt được ý nghĩa tương đối** (capture the relative meaning) của các từ.
- **Sự phân cụm (Clustering):** Các từ xuất hiện ở những nơi (ngữ cảnh) tương tự sẽ được mã hóa theo cách tương tự. Bạn có thể tận dụng điều này để tìm các mẫu (ví dụ: các từ gần nhất với "bác sĩ" (doctor) có thể là "y tá" (nurse), "bác sĩ phẫu thuật" (surgeon)...).
- Để chúng ta biết mối quan hệ không xác định giữa các từ, -> bằng cách sử dụng các mối quan hệ đã biết giữa những người khác. 

---
### **Visualization and PCA**
---

- Thường thì bạn sẽ có các **vectơ ở kích thước rất cao** (high dimensions). Bạn muốn **giảm chiều** (reduce the dimension) của chúng xuống **hai chiều** (two dimensions) để có thể vẽ (plot) trên trục XY.
- Bạn sẽ học về **Phân tích thành phần chính (Principal Component Analysis - PCA)**, thuật toán cho phép bạn làm điều này.
- PCA được sử dụng để **hình dung (visualize)** các biểu diễn vectơ có kích thước cao.

#### Động lực (Trực quan hóa)
> Biểu đồ trực quan hóa

![12_Example_Visualization](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/12_Example_Visualization.png)

- Giả sử bạn có biểu diễn vectơ trong không gian chiều cao, và bạn biết rằng các từ "dầu khí" (oil and gas) và "thành phố và thị trấn" (city and town) có liên quan.
- Bạn muốn xem liệu biểu diễn của mình có nắm bắt được mối quan hệ đó hay không.
- **Giảm kích thước** (Dimensionality reduction) là lựa chọn hoàn hảo cho nhiệm vụ này.
- Bạn có thể sử dụng PCA để lấy biểu diễn trong không gian có **ít chiều hơn** (fewer dimensions) (ví dụ: ba tính năng trở xuống).
- Nếu bạn nhận được **biểu diễn hai chiều**, bạn có thể vẽ hình ảnh của các từ.
- Trong hình ảnh đó, bạn có thể thấy liệu các từ liên quan (như "dầu khí" và "thành phố và thị trấn") có được **tập hợp (clustered)** lại với nhau hay không.
- Bạn thậm chí có thể tìm thấy các mối quan hệ khác mà bạn không mong đợi.


#### Cách thức hoạt động (Tổng quan)

- Để đơn giản, hãy xét một không gian **hai chiều** (2D) mà bạn muốn giảm xuống **một tính năng** (1D).
- Đầu tiên, PCA sẽ tìm một tập hợp các **tính năng không tương quan (uncorrelated features)**.
- Sau đó, nó **chiếu (project)** dữ liệu của bạn vào không gian một chiều, cố gắng **giữ lại càng nhiều thông tin càng tốt (retain as much information as possible)**.

#### Tóm tắt

- PCA là một thuật toán **giảm kích thước** (dimensionality reduction) có thể tìm thấy các **tính năng không tương quan**.
- Nó rất hữu ích cho việc **trực quan hóa dữ liệu** (visualizing your data).
- Nó cho phép bạn biến một **vectơ chiều d** (d-dimensional vector) thành hai chiều để tạo ra một **biểu đồ** (plot).

---
### **PCA Algorithmn**
---

- Phần này nói về **giá trị riêng (eigenvalues)** và **vectơ riêng (eigenvectors)**, và cách sử dụng chúng để **giảm kích thước (reduce the dimension)** của các tính năng.
- Mục tiêu là có được các **tính năng không tương quan (uncorrelated features)** và **giữ càng nhiều thông tin càng tốt** (keep as much information as possible) từ việc nhúng ban đầu.

> Giảm chiều bằng PCA 

![13_Visualization_PCA](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/13_Visualization_PCA.png)


- Quy trình giảm kích thước bằng **PCA** (Phân tích thành phần chính):
    1.  Bắt đầu với không gian vectơ ban đầu.
    2.  Lấy các tính năng không tương quan cho dữ liệu.
    3.  **Chiếu (project)** dữ liệu vào một số tính năng mong muốn, giữ lại nhiều thông tin nhất.
- Trong PCA, **Eigenvector** (vectơ riêng) của **ma trận đồng phương sai (covariance matrix)** từ dữ liệu của bạn cung cấp **hướng (direction)** của các tính năng không tương quan. -> Giải pháp kiếm uncorrelated features
- **Eigenvalues** (giá trị riêng) là **biến thể (variance)** của tập dữ liệu trong mỗi tính năng mới đó.
- Để thực hiện PCA, bạn cần lấy Eigenvector và Eigenvalues từ ma trận phương sai đồng của dữ liệu. -> Giải pháp cho việc giữ càng nhiều thông tin càng tốt. 

> Thuật toán PCA

![14_PCA_Algorithmn](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/14_PCA_Algorithmn.png)


- **Bước 1: Lấy các tính năng không tương quan.**
    + **Bình thường hóa (normalize)** dữ liệu.
    + Lấy **ma trận hiệp phương sai (covariance matrix)**.
    + Thực hiện **Phân hủy giá trị đơn lẻ (Singular Value Decomposition - SVD)**.
    + SVD trả về ba ma trận: Ma trận đầu tiên (ký hiệu là **U**) chứa các **Eigenvector** (xếp chồng lên nhau theo cột), và ma trận thứ hai (ký hiệu là **S**) có các **Eigenvalue** trên đường chéo.
    + (SVD đã được triển khai trong nhiều thư viện lập trình).
- **Bước 2: Chiếu dữ liệu (Project the data).**
    + Sử dụng Eigenvector (U) và Eigenvalue (S).
    + Thực hiện **tích chấm (dot product)** giữa ma trận chứa các **nhúng từ (word embeddings)** của bạn và **N cột đầu tiên (first N columns)** của ma trận U.
    + **N** là số chiều bạn muốn có ở cuối (ví dụ: hai chiều để **hình dung - visualization**).
- **Lưu ý quan trọng:** Các Eigenvector và Eigenvalue phải được **sắp xếp (sorted)** theo Eigenvalue theo **thứ tự giảm dần (descending order)** để đảm bảo giữ lại nhiều thông tin nhất (hầu hết các thư viện tự động làm điều này).
- **Tóm lại:** 
    + Eigenvector cho hướng của các tính năng không tương quan
    + Eigenvalue cho biết biến thể. 
    + (Dot product) Tích chấm sẽ chiếu dữ liệu lên một không gian vectơ mới.

- **Các bước để tính toán PCA:**

    + Chuẩn hóa trung bình (Mean normalize) dữ liệu của bạn.
    + Tính toán ma trận hiệp phương sai (covariance matrix).
    + Tính toán SVD (Phân rã giá trị suy biến) trên ma trận hiệp phương sai của bạn. Phép tính này trả về $[U S V] = svd(\Sigma)$. Ba ma trận $U$, $S$, $V$ được vẽ ở trên. $U$ được gán nhãn là vector riêng (eigenvectors), và $S$ được gán nhãn là giá trị riêng (eigenvalues).
    + Sau đó, bạn có thể sử dụng $n$ cột đầu tiên của vector $U$, để lấy dữ liệu mới bằng cách nhân $XU[:, 0:n]$.

---
### **The Rotation Matrix (Optional Reading)**
---

#### **Phép quay ngược chiều kim đồng hồ (Counterclockwise Rotation)**

- Nếu bạn muốn quay một vector $r$ với tọa độ $(x, y)$ và góc $\alpha$ ngược chiều kim đồng hồ một góc $\beta$ để được vector $r'$ với tọa độ $(x', y')$ thì ta có:

$$x = r * \cos(\alpha)$$
$$y = r * \sin(\alpha)$$
$$x' = r' * \cos(\alpha + \beta)$$
$$y' = r' * \sin(\alpha + \beta)$$
- Phép cộng lượng giác cho ta:

$$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

$$\sin(\alpha + \beta) = \cos(\alpha)\sin(\beta) + \sin(\alpha)\cos(\beta)$$
- Để xem chứng minh, hãy xem [phần trang Wikipedia này](https://en.wikipedia.org/wiki/Proofs_of_trigonometric_identities#Angle_sum_identities).



- [Chứng minh: Công thức cộng góc Lượng giác (SIN và COS)](https://www.youtube.com/watch?v=i_F-s2G-xDc)
Video này giải thích cách chứng minh các công thức cộng góc cho sin và cosin, vốn là các công thức được đề cập trong ảnh của bạn.

- Vì độ dài của vector không đổi,

$$x' = r * \cos(\alpha)\cos(\beta) - r * \sin(\alpha)\sin(\beta)$$
$$y' = r * \cos(\alpha)\sin(\beta) + r * \sin(\alpha)\cos(\beta)$$

- Điều này tương đương với:

$$x' = x * \cos(\beta) - y * \sin(\beta)$$
$$y' = x * \sin(\beta) + y * \cos(\beta)$$

- Viết dưới dạng phép nhân ma trận với **vector hàng**, ta có:

$$[x', y'] = [x, y] \cdot \begin{bmatrix} \cos(\beta) & \sin(\beta) \\ -\sin(\beta) & \cos(\beta) \end{bmatrix}$$

- với ma trận quay bằng,

$$R = \begin{bmatrix} \cos(\beta) & \sin(\beta) \\ -\sin(\beta) & \cos(\beta) \end{bmatrix}$$

- Viết dưới dạng phép nhân ma trận với **vector cột**, ta có:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos(\beta) & -\sin(\beta) \\ \sin(\beta) & \cos(\beta) \end{bmatrix} \cdot \begin{bmatrix} x \\ y \end{bmatrix}$$

- với ma trận quay bằng,

$$R = \begin{bmatrix} \cos(\beta) & -\sin(\beta) \\ \sin(\beta) & \cos(\beta) \end{bmatrix}$$

- Lưu ý rằng vị trí của $-\sin(\beta)$ trong ma trận quay đã thay đổi.

#### **Phép quay cùng chiều kim đồng hồ (Clockwise Rotation)**

- Nếu phép quay là cùng chiều kim đồng hồ, thì ma trận quay để nhân với **vector hàng** trở thành,

$$R = \begin{bmatrix} \cos(-\beta) & \sin(-\beta) \\ -\sin(-\beta) & \cos(-\beta) \end{bmatrix}$$

- Vì $\sin(-\beta) = -\sin(\beta)$ và $\cos(-\beta) = \cos(\beta)$

- điều này tương đương với

$$R = \begin{bmatrix} \cos(\beta) & -\sin(\beta) \\ \sin(\beta) & \cos(\beta) \end{bmatrix}$$

- Vì vậy, phép quay cùng chiều kim đồng hồ của một vector $[x, y]$ có thể được biểu diễn là,

$$[x', y'] = [x, y] \cdot \begin{bmatrix} \cos(\beta) & -\sin(\beta) \\ \sin(\beta) & \cos(\beta) \end{bmatrix}$$

- Ma trận quay để nhân với **vector cột** trở thành,

$$R = \begin{bmatrix} \cos(-\beta) & -\sin(-\beta) \\ \sin(-\beta) & \cos(-\beta) \end{bmatrix}$$

- tương đương với,

$$R = \begin{bmatrix} \cos(\beta) & \sin(\beta) \\ -\sin(\beta) & \cos(\beta) \end{bmatrix}$$
- Vì vậy, phép quay cùng chiều kim đồng hồ của một vector 

$$\begin{bmatrix} x \\ y \end{bmatrix}$$

- có thể được biểu diễn là,

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos(\beta) & \sin(\beta) \\ -\sin(\beta) & \cos(\beta) \end{bmatrix} \cdot \begin{bmatrix} x \\ y \end{bmatrix}$$

**Tác giả:** Reinoud Bosch

### **Giải một số bài tập để hiểu cách tính Euclidian và Cosine Similarity**

> Câu 1: Tính Euclidian

![15_Câu_1](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/15_Câu_1.png)


Tất nhiên, đây là cách tính chi tiết.

Đáp án **5.91608** là chính xác.

#### 📐 Công thức Khoảng cách Euclidean (Euclidean Distance)

Khoảng cách Euclidean giữa hai vector $A = (A_1, A_2, A_3)$ và $B = (B_1, B_2, B_3)$ được tính bằng công thức:

$$d(A, B) = \sqrt{(B_1 - A_1)^2 + (B_2 - A_2)^2 + (B_3 - A_3)^2}$$

Nói đơn giản, đó là:
1.  Tính **hiệu** ở từng chiều.
2.  **Bình phương** các kết quả hiệu đó.
3.  **Cộng** tất cả lại.
4.  Lấy **căn bậc hai** của tổng cuối cùng.

---

#### 📝 Các bước tính toán

Với hai vector của bạn:
* $A = (1, 2, 3)$
* $B = (4, 7, 2)$

**1. Tính hiệu (B - A):**
* $B_1 - A_1 = 4 - 1 = 3$
* $B_2 - A_2 = 7 - 2 = 5$
* $B_3 - A_3 = 2 - 3 = -1$

**2. Bình phương các hiệu:**
* $(3)^2 = 9$
* $(5)^2 = 25$
* $(-1)^2 = 1$

**3. Cộng các kết quả bình phương:**
$$9 + 25 + 1 = 35$$

**4. Lấy căn bậc hai:**
$$d(A, B) = \sqrt{35}$$

Khi tính giá trị thập phân:
$$\sqrt{35} \approx 5.916079...$$

Giá trị này khớp với đáp án được chọn là **5.91608**.

> Câu 2: Tính Cosine Similarity

![16_Câu_2](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W3/16_Câu_2.png)

- Hướng dẫn giải quyết bài toán

Chào bạn, hướng dẫn trong ô màu xanh là hoàn toàn chính xác.

Bài toán này dựa trên một khái niệm phổ biến trong xử lý ngôn ngữ tự nhiên gọi là "vector analogy" (tương tự vector). Ý tưởng là mối quan hệ giữa một quốc gia và thủ đô của nó có thể được biểu diễn bằng một vector.

Mối quan hệ này có thể được tính bằng:
$V_{\text{quan hệ}} = \text{Vector}(\text{USA}) - \text{Vector}(\text{Washington})$

Chúng ta có thể áp dụng vector quan hệ này cho một thủ đô khác (Ankara) để tìm ra quốc gia tương ứng:
$\text{Vector}(\text{Quốc gia cần tìm}) \approx V_{\text{quan hệ}} + \text{Vector}(\text{Ankara})$

Kết hợp lại, chúng ta có công thức mà đề bài gợi ý:
**$\text{Vector}(\text{Quốc gia cần tìm}) = (\text{USA} - \text{Washington}) + \text{Ankara}$**

Sau khi tính ra "Vector quốc gia cần tìm" này, chúng ta sẽ dùng **cosine similarity** để so sánh nó với các vector quốc gia đã cho ($Turkey$, $Russia$, $Japan$) và chọn quốc gia có độ tương đồng cao nhất.

---

#### 1. Tính "Vector Quốc gia cần tìm" (Target Vector)

Hãy gọi vector này là $V_{\text{target}}$.

* $USA = (5, 6)$
* $Washington = (10, 5)$
* $Ankara = (9, 1)$

$$V_{\text{target}} = ( (5, 6) - (10, 5) ) + (9, 1)$$

* **Bước 1: Trừ vector**
    $$(5, 6) - (10, 5) = (5 - 10, 6 - 5) = (-5, 1)$$

* **Bước 2: Cộng vector**
    $$V_{\text{target}} = (-5, 1) + (9, 1) = (-5 + 9, 1 + 1) = (4, 2)$$

Vậy, vector chúng ta cần tìm là **$V_{\text{target}} = (4, 2)$**.

---

#### 2. So sánh $V_{\text{target}}$ với các Quốc gia

Bây giờ, chúng ta sẽ tính cosine similarity giữa $V_{\text{target}} = (4, 2)$ với từng vector quốc gia.

Công thức: $\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

* **Độ lớn của $V_{\text{target}} = (4, 2)$:**
    $\|V_{\text{target}}\| = \sqrt{4^2 + 2^2} = \sqrt{16 + 4} = \sqrt{20}$

##### A. So sánh với Turkey = (3, 1)
* **Tích vô hướng:** $(4 \times 3) + (2 \times 1) = 12 + 2 = 14$
* **Độ lớn:** $\|Turkey\| = \sqrt{3^2 + 1^2} = \sqrt{9 + 1} = \sqrt{10}$
* **Cosine Similarity:**
    $$\cos(\theta) = \frac{14}{\sqrt{20} \times \sqrt{10}} = \frac{14}{\sqrt{200}} \approx 0.9899$$

##### B. So sánh với Russia = (5, 5)
* **Tích vô hướng:** $(4 \times 5) + (2 \times 5) = 20 + 10 = 30$
* **Độ lớn:** $\|Russia\| = \sqrt{5^2 + 5^2} = \sqrt{25 + 25} = \sqrt{50}$
* **Cosine Similarity:**
    $$\cos(\theta) = \frac{30}{\sqrt{20} \times \sqrt{50}} = \frac{30}{\sqrt{1000}} \approx 0.9487$$

##### C. So sánh với Japan = (4, 3)
* **Tích vô hướng:** $(4 \times 4) + (2 \times 3) = 16 + 6 = 22$
* **Độ lớn:** $\|Japan\| = \sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = 5$
* **Cosine Similarity:**
    $$\cos(\theta) = \frac{22}{\sqrt{20} \times 5} = \frac{22}{\sqrt{20} \times \sqrt{25}} = \frac{22}{\sqrt{500}} \approx 0.9839$$

---

#### 3. Kết luận

So sánh các điểm tương đồng:
* **Turkey:** 0.9899 (Cao nhất)
* **Japan:** 0.9839
* **Russia:** 0.9487

Vector của **Turkey** có độ tương đồng cosine cao nhất với vector (USA - Washington) + Ankara. Do đó, đáp án chính xác là **Turkey**.