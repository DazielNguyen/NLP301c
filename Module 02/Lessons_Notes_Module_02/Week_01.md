# **Module 02 - Natural Language Processing with Probabilistic Models**
## **Week 1: Autocorrect and Minimum Edit Distance** -> Consist of 2 small labs and 1 Asgm
---
### **Overview**
---
#### Mục tiêu của tuần 2 
- Hiểu được autocorrect là gì? 
- Xây dựng được model để tự động sửa lỗi chính tả từ 
- Hiểu được về chỉnh sửa khoảng cách tối thiểu giữa các chuỗi từ (Minimum edit distance)
- Lập trình động (Dynamic Programming) dùng các thuật toán chính sửa khoảng cách tối thiểu -> Giải quyết các vấn đề về tối ưu hóa 
> Overview Tuần 2 

![01_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/01_Overview.png)

---
### **Autocorrect (Tự động sửa lỗi)**
---
- Autocorrect là một ứng dụng thay đổi các từ sai chính tả thành các từ chính xác
- Autocorrect hoạt động như thế nào? 

    1. Xác định một từ không chính xác. Như sai chính tả....
    2. Tìm các chuỗi từ 1 đến n để chỉnh sửa khoảng cách xa
    3. Lọc các chuỗi từ thực được đánh vần chính tả chính xác
    4. Tính toán xác suất từ, cho biết khả năng mỗi từ xuất hiện trong ngữ cảnh khác nhau và chọn ứng cử viên có khả năng thay thế nhất.

---
### **Building the model**
---

#### Bước 1: Xác định từ sai chính tả (Identify the misspelled word)

- **Làm thế nào?** Kiểm tra xem một từ có tồn tại trong **từ điển** (dictionary) hay không.
- Nếu từ đó **không có** trong từ điển, nó sẽ bị đánh dấu là sai chính tả và cần sửa.
- **Lưu ý:** Phương pháp này chỉ xác định **lỗi chính tả** (spelling errors), không phải **lỗi ngữ cảnh** (contextual errors). (Ví dụ: "hươu" (deer) là một từ đúng chính tả, nó sẽ vượt qua bộ lọc ngay cả khi ngữ cảnh có thể sai).


#### Bước 2: Tìm các chuỗi trong khoảng cách n chỉnh sửa (Find strings n edit distance away)

- **Chỉnh sửa (Edit):** Là một thao tác (operation) được thực hiện trên một chuỗi để thay đổi nó.
- **Khoảng cách chỉnh sửa n (n edit distance):** Là số lượng *n* thao tác cần thiết để biến một chuỗi thành chuỗi khác.
- Bốn loại chỉnh sửa cơ bản được sử dụng:

    1.  **Chèn (Insert):** Thêm một chữ cái vào bất kỳ vị trí nào.
        * *Ví dụ:* `to` $\rightarrow$ `top` (chèn 'p') hoặc `two` (chèn 'w').
    2.  **Xóa (Delete):** Loại bỏ một chữ cái.
        * *Ví dụ:* `hat` $\rightarrow$ `ha` (xóa 't'), `at` (xóa 'h'), `ht` (xóa 'a').
    3.  **Hoán đổi (Switch):** Hoán đổi hai chữ cái **liền kề** (adjacent).
        * *Ví dụ:* `eta` $\rightarrow$ `eat` (hoán đổi 't' và 'a') hoặc `tea` (hoán đổi 'e' và 't').
        * *Lưu ý:* Điều này *không* bao gồm các chữ cái không liền kề (ví dụ: `eta` $\rightarrow$ `ate` không phải là một lần "hoán đổi").
    4.  **Thay thế (Replace):** Thay đổi một chữ cái này thành một chữ cái khác.
        * *Ví dụ:* `jaw` $\rightarrow$ `jar` (thay 'w' bằng 'r').

- Đối với tự động sửa lỗi, *n* thường là 1 đến 3. Bằng cách kết hợp các chỉnh sửa này, bạn tạo ra một danh sách tất cả các chuỗi có thể có trong vòng *n* khoảng cách chỉnh sửa.


#### Bước 3: Lọc ứng cử viên (Filter candidates)

- Nhiều chuỗi được tạo ra ở Bước 2 không phải là từ có thật.
- **Làm thế nào?** Bạn **lọc (filter)** danh sách ứng cử viên bằng cách kiểm tra lại chúng với **từ điển** (giống như ở Bước 1).
- Nếu một chuỗi ứng cử viên **không** xuất hiện trong từ điển, nó sẽ bị **xóa (remove)** khỏi danh sách.
- Sau bước này, bạn chỉ còn lại một danh sách các từ có thật, đúng chính tả.

#### Tóm tắt

- Ba bước đầu tiên: 
    + (1) Xác định lỗi
    + (2) Tạo các chỉnh sửa, và 
    + (3) Lọc các ứng cử viên. Bước thứ tư và cũng là bước cuối cùng—**tính toán xác suất**

---
### **Building the model II + Bổ sung thêm Bước 4**
---

- Cách tính **xác suất** (probability) của từng từ ứng cử viên chính xác.
- Đây là **Bước 4** (bước cuối cùng): Tính xác suất giá trị (probability values) và tìm từ có **khả năng nhất** (most likely) từ danh sách ứng cử viên.
- Ví dụ: Từ "and" phổ biến hơn "an" trong một **kho tài liệu** (corpus). Đây là cách tự động sửa (auto-correct) biết nên chọn từ nào.

> Ví dụ về cách tính xác suất từ 

![02_Example_Calculating_Word_Prob](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/02_Example_Calculating_Word_Prob.png)

#### Cách tính Xác suất

1.  **Tính Tần số:**
    - Để tính xác suất, trước tiên bạn cần **tần số từ** (word frequencies) và **tổng số từ** (total number of words) trong kho tài liệu.
    - (Một kho tài liệu thường rất lớn, như tất cả các cuốn sách Harry Potter).
2.  **Ví dụ đơn giản:**
    - Script sử dụng kho tài liệu (corpus) được định nghĩa là một câu duy nhất: "Tôi hạnh phúc vì tôi đang học hỏi."
    - Ví dụ về đếm (từ script): Từ "I" xuất hiện 2 lần, từ "am" cũng xuất hiện 2 lần.
    - Tổng số từ trong kho này là 7.
3.  **Công thức Xác suất:**
    - Xác suất của một từ = (Số lần từ xuất hiện) / (Tổng số từ trong kho).
    - Ví dụ (từ script): $P(\text{am}) = 2 / 7$.
4.  **Lựa chọn:** Tự động sửa sẽ tìm và chọn từ ứng cử viên có **xác suất cao nhất** (highest probability) làm từ thay thế.


#### Tóm tắt toàn bộ 4 bước

Script tóm tắt lại toàn bộ quy trình tự động sửa lỗi (ví dụ: sửa từ "deah"):

1.  **Xác định** (Identify) từ ("deah") là sai chính tả (bằng cách kiểm tra nó so với các từ đã biết/từ điển).
2.  **Tạo** (Generate) một danh sách tất cả các chuỗi cách *n* **chỉnh sửa** (edits) (hay *n* **khoảng cách chỉnh sửa** - edit distance).
3.  **Lọc** (Filter) danh sách này để chỉ bao gồm những từ có thật (có trong từ điển).
4.  **Tính toán xác suất** (Calculate probabilities) cho mỗi từ còn lại và chọn từ có xác suất cao nhất làm thay thế.

#### **Kết luận:** 
> Tổng quan về các bước xây dựng mô hình Autocorrect

![03_Step_by_Step_Built_Autocorrect_Model](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/03_Step_by_Step_Built_Autocorrect_Model.png)

- Bạn đã học về **chỉnh sửa** (edits) và **khoảng cách chỉnh sửa** (edit distance) và cách chúng đo lường **sự tương đồng** (similarity) giữa các từ.
- Bạn đang lưu trữ **số lượng từ (count of words)**, và sau đó bạn có thể sử dụng chúng để tạo ra **xác suất (probabilities)**.
- Trong tuần này, bạn sẽ chỉ đếm **xác suất của các từ (đơn lẻ) xuất hiện**.
- Nếu bạn muốn xây dựng một hệ thống **tự động sửa lỗi (auto-correct) tinh vi hơn**, bạn có thể theo dõi **hai từ xuất hiện liền kề nhau**.
- Điều này cho phép bạn sử dụng từ đứng trước để quyết định (ví dụ: tổ hợp "there friend" hay "their friend" có khả năng cao hơn).
- Tuy nhiên, trong tuần này, bạn sẽ chỉ triển khai xác suất bằng cách sử dụng **tần suất từ (đơn lẻ)**.

---
### **Minimum edit distance**
---
> Làm thế nào để đánh giá giữa 2 chuỗi có điểm tương đồng?

> Số lượng tối thiểu của chỉnh sửa cần thiết để chuyển đổi 1 chuỗi sang một chuỗi khác. 

- **Chủ đề:** **Khoảng cách chỉnh sửa tối thiểu** (Minimum Edit Distance).
- **Định nghĩa:** Đây là **số phép toán (operations) thấp nhất** cần thiết để biến đổi một chuỗi thành chuỗi kia.
- **Ứng dụng:** Nó có nhiều ứng dụng trong NLP (như **chỉnh sửa chính tả**, **tương tự tài liệu**, **dịch máy**) và trong sinh học tính toán (như **giải trình tự DNA**).
- **Các loại thao tác (Chỉnh sửa):** Bạn sẽ sử dụng ba thao tác bạn đã biết: **chèn** (insert), **xóa** (delete), và **thay thế** (replace).
- **Ví dụ (Chi phí bằng nhau):** Để biến "play" thành "stay", bạn cần 2 lần "thay thế" (p $\rightarrow$ s và l $\rightarrow$ t). Tổng số chỉnh sửa là 2.

> Ví dụ cụ thể nơi chúng ta tính toán chi phí (tức là khoảng cách chỉnh sửa) giữa hai chuỗi.

![04_Minium_edit_distance](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/04_Minium_edit_distance.png)


- **Khái niệm mới: Chi phí Chỉnh sửa (Edit Costs):**
    + Thay vì mọi hoạt động đều có chi phí (cost) là 1, bây giờ bạn sẽ xem xét các chi phí khác nhau cho mỗi loại.
    + Mục tiêu là giảm thiểu **tổng chi phí chỉnh sửa** (total edit cost).
    + **Ví dụ về chi phí:** Chèn = 1, Xóa = 1, **Thay thế = 2**.
    + Lý do (trực quan): "Thay thế" giống như một thao tác "Xóa" theo sau là một thao tác "Chèn".
- **Ví dụ (Với chi phí mới):** Để biến "play" thành "stay" (2 lần thay thế), tổng chi phí bây giờ là 2 + 2 = 4.

- **Vấn đề với chuỗi dài:**
    + Bạn có thể tìm ra khoảng cách chỉnh sửa cho các ví dụ đơn giản "chỉ bằng cách nhìn vào nó".
    + Tuy nhiên, đối với các chuỗi dài (văn bản lớn, DNA), việc thử **"lực thô"** (brute force) (liệt kê tất cả các khả năng) sẽ mất thời gian rất, rất dài.
    + Độ phức tạp tính toán **tăng theo cấp số nhân** (increases exponentially).
- **Giải pháp (Cách nhanh hơn):**
    + Sử dụng **cách tiếp cận dạng bảng** (tabular approach) để tăng tốc độ.
    + Phương pháp này sử dụng một khái niệm mới gọi là **lập trình động** (dynamic programming).

---
### **Minimum edit distance algorithmn**
---
> Khi tính toán khoảng cách chỉnh sửa tối thiểu, bạn sẽ bắt đầu với một **source word** và chuyển đổi nó thành **target word**. 

> Ví dụ 

![05_Example_MED_Algorithmn](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/04_Minium_edit_distance.png)

- Để đi từ **#** $\rightarrow$ **#** bạn cần một chi phí là **0**
- Từ **p** $\rightarrow$ **#** bạn nhận **1**, bởi vì đó là chi phí của 1 lần **Delete**
- **p** $\rightarrow$ **s** là 2, vì đó là chi phí tổi thiểu có thể dùng để đi từ **p** đến **s**
- Bạn có thể tiếp tục theo cách này bằng cách điền từng phần tử một, nhưng hóa ra có một cách nhanh hơn để làm việc này.

---
### **Minimum edit distance algorithmn II**
---
> Để điền vào bảng trống bên dưới 

![06_Example_MED_Algorithmn_II](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/06_Example_MED_Algorithmn_II.png)

> Có 3 công thức tính như sau: 

- **D[i,j] = D[i-1,j] + del_cost**: Điều này chỉ ra rằng bạn muốn điền vào ô hiện tại (i,j) bằng cách sử dụng chi phí trong ô ngay phía trên.
- **D[i,j] = D[i,j-1] + ins_cost**: Điều này cho thấy bạn muốn điền giá trị vào ô hiện tại (i,j) bằng cách sử dụng chi phí trong ô nằm ngay bên trái nó.
- **D[i,j] = D[i-1,j-1] + rep_cost**: Chi phí thay thế có thể là 2 hoặc 0 tùy thuộc vào việc bạn có thực sự thay thế nó hay không.

> Ở mỗi bước thời gian, bạn kiểm tra ba đường có thể mà bạn có thể đến từ đó và chọn đường ít tốn kém nhất. Khi bạn hoàn tất, bạn sẽ nhận được kết quả sau:

![07_Example_MED_Algorithmn_II](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2002/Image_Module_02/M2_W1/07_Example_MED_Algorithmn_II.png)

---
### **Minimum edit distance algorithmn III**
---



---
### **Minimum edit distance III**
---


