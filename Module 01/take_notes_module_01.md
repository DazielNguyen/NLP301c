# **Module 01** - Natural Language Processing with Classification and Vector Spaces
## Week 1: Sentiment Analysis with Logistic Regression

- Biểu diễn văn bản dưới dạng text thành vector và xây dựng một bộ phân loại sẽ phân loại văn bản mẫu thành hai loại (Tâm lý tích cực hoặc Tâm lý tiêu cực). Sử dụng Logistic Regression. 

### Logistic Regression
#### 1. Supervised Machine Learning (Học có giám sát)

![M1_W1_01_Supervised ML](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_01_Supervised%20ML.png)
> Kiến trúc của Supervise ML

- Trong máy học giám sát bạn có các tính năng đầu vào **X** và tập hợp các nhãn **Y**
- Để đảm bảo rằng bạn nhận được **dự đoán chính xác nhất** dựa trên dữ liệu của bạn.
- Mục tiêu của bản là **giảm thiểu** tỷ lệ **lỗi** của bạn hoặc **chi phí** càng nhiều càng tốt. 
- Và để làm được điều này, phải chạy được **Prediction Function** của bạn, cái mà lấy trong dữ liệu tham số để gán các Feature của bạn để đầu ra nhãn Y^
- Mapping tốt nhất từ các Features đến nhãn đạt được khi sự khác biệt giữa các giá trị kỳ vọng Y và giá trị dự đoán Y^ được **giảm thiểu**.
- **Hàm chi phí** thực hiện bằng cách so sánh mức độ gần gũi giữa Output Y^ của bạn với nhãn Y.
- Sau đó bạn có thể cập nhật tham số và lặp lại toàn bộ quá trình xử lý cho đến khi tối ưu được chi phí thấp nhất. 



#### 2. Sentiment Analysis (Phân tích tình cảm)

![Sentiment analysis Ví dụ](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_02_Sentiment%20analysis.png)
>Ví dụ về Sentiment analysis 

- Trong ví dụ này bạn có: 

    ```
    Tweet: Tôi rất hạnh phúc bởi vì tôi đang học NLP
    ```

- Và mình cần chứng minh rằng câu nói này đang là **tâm lý tích cực** hay **tâm lý tiêu cực**
- Để thực hiện bạn cần chuẩn bị tập training: 

    ```
    Postive (Tâm lý tích cực): -> Lable: 1
    Negative (Tâm lý tiêu cực): -> Lable: 0
    ```

- Dùng Logistic Regression đã dán nhãn, gán quan sát của nó cho hai lớp khác biệt

![Cách xây dựng Sentiment analysis](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_02_Sentiment%20analysis.png)
> Cách xây dựng Sentiment analysis

- Để xây dựng một bộ phân loại hồi quy logistic, có khả năng dự đoán tình cảm của một tweet tùy ý. 
- Xử lý các tweet thô trong training set và trích xuất các tính năng hữu ích
- Sau đó train Logistic Regression của bạn cùng với đó phải giảm thiểu chi phí. 
- Classify, cuối cùng bạn sẽ có thể đưa ra dự đoán của bạn

### Vocabulary & Feature Extraction
#### 3. Vocabulary 
- Để biểu diễn text dưới dạng vector, cần phải xây dựng một bộ từ vựng và nó sẽ cho phép bạn mã hóa bất kỳ text nào hoặc bất kỳ tweet nào dưới dạng một mảng số.

![M1_01_01_Vocabulary](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_03_Vocabulary.png)

> Cách lưu các text thành mảng số

- Hình ảnh một danh sách các tweet, trực quan hóa nó sẽ là các câu. 
- Sau đó từ vựng V sẽ là danh sách các từ duy nhất trong danh sách các tweet của bạn. 
- Để có được list đó thì bạn phải xem qua tất cả các từ vựng từ tất cả các tweet của bạn và lưu mọi từ mới xuất hiện trong tìm kiếm của bạn. 
- Lưu ý trong 2 câu có lặp từ thì chỉ lấy 1 từ duy nhất, không lặp lại hai từ đó. 

#### 4. Feature Extraction

![Biểu diễn Feature Extraction](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_04_Feature%20Extraction.png)
> Giải thích ảnh trên: 

- Chúng ta có một câu (I am happy because I am learning NLP)
- Làm sao để xác định câu này có trong list. 
- Thì nó sẽ gán giá trị là 1 nếu từ vựng trong câu trên có xuất hiện trong list, còn tất cả các giá trị còn lại không xuất hiện trong list sẽ hiểu là 0
- Những nó sẽ sinh ra vấn đề -> **Quá nhiều số 0 và biễu diễn quá thưa thớt.**

#### 5. Vấn đề biểu diễn thưa thớt 

![M1_W1_05_Problem Spare Representation](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_05_Problem%20Spare%20Representation.png)
> Giải thích ảnh trên: 

- Với sử biểu diễn thưa thớt, thì mô hình hồi quy Logistic sẽ phải học **n + 1 parameter**. 
- Trong đó n sẽ bằng kích thước từ vựng -> Nếu **kích thước từ vựng lớn** điều này sẽ là 1 vấn đề. 
- Mô hình mất rất nhiều thời gian để training và mất nhiều thời gian hơn cần thiết để đưa ra dự đoán

- Với text đã được học cách biểu diễn dứi dạng một Vector có kích thước v cụ thể -> 1 Tweet có thể xây dựng dưới một từ vựng của chiều V. 
- Nếu V trở nên lớn hơn -> **bạn sẽ gặp phải một số vấn đề**
- Như bạn có thể thấy, khi V trở nên lớn hơn, vector trở nên thưa thớt hơn. Hơn nữa, chúng ta sẽ có nhiều đặc trưng hơn và kết quả là phải **huấn luyện nhiều tham số θ** của V hơn. 
- Điều này có thể dẫn đến **thời gian huấn luyện lâu hơn** và **thời gian dự đoán cũng lớn hơn**.

**Negative and Positive** 

#### 6. Feature Extraction with Frequencies

- Mỗi hàng text sẽ là 1 Tweet 

- Cho một tập hợp dữ liệu với các tweet tích cực và tiêu cực như sau

![M1_W1_06_Feature Extraction with Frequencies_01](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_06_Feature%20Extraction%20with%20Frequencies_01.png)

- Bạn phải mã hóa mỗi tweet dưới dạng một vectơ. 
- Trước đây, vectơ này có kích thước V. 
- Bây giờ, như bạn sẽ thấy trong các video sắp tới, bạn sẽ biểu diễn nó bằng một vectơ có kích thước 3. 
- Để làm được điều này, bạn phải tạo một từ điển để gán từ và lớp mà nó xuất hiện (tích cực hoặc tiêu cực) với số lần từ đó xuất hiện trong lớp tương ứng của nó.

![M1_W1_06_Feature Extraction with Frequencies_02](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_06_Feature%20Extraction%20with%20Frequencies_02.png)

- Trong hai video trước, chúng tôi gọi từ điển này là `freqs`. 
- Trong bảng trên, bạn có thể thấy các từ như happy và sad có xu hướng nghiêng về một thái cực rõ ràng, trong khi các từ khác như "I, am" thường có xu hướng trung lập hơn. 
- Dựa trên từ điển này và tweet, "I am sad, I am not learning NLP", bạn có thể tạo một vector tương ứng với đặc trưng như sau:

![M1_W1_06_Feature Extraction with Frequencies_03](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_06_Feature%20Extraction%20with%20Frequencies_03.png)

- Để mã hóa đặc điểm tiêu cực, bạn có thể làm việc tương tự

![M1_W1_06_Feature Extraction with Frequencies_04](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_06_Feature%20Extraction%20with%20Frequencies_04.png)

- Do đó, bạn sẽ nhận được vectơ đặc trưng sau đây[1,8,11] . 1, tương ứng với độ lệch (bias), 8 là đặc trưng dương, và 11 là đặc trưng âm.

#### 7. Preprocessing
- Học về hai khái niệm chính về tiền xử lý: 
    + Khái niệm thứ nhất gọi là `steaming` (**bắt nguồn từ**) 
    + Khái niệm thứ hai gọi là `stop word` (**dừng từ**)
- Học về cách sử dụng `steaming` và `stop word` để xử lý trước văn bản

- Khi tiền xử lý, bạn phải thực hiện theo các bước sau: 
    + Loại bỏ các handle và URL
    + Phân tách chuỗi thành các từ
    + Xóa các `stop word` như: "and, is, a, o, v.v"
    + `steaming` hoặc chuyển đổi từng từ thành gốc của nó. Ví dụ như dancer, dancing, danced, thành **danc**. Bạn có thế sử dụng `porter stemmer` để xử lý việc này. 
    + Chuyển đổi tất cả các từ của bạn sang **chữ thường**. 

- Ví dụ như dòng tweet sau
   
    ```
    "@YMourri and @AndrewYNg are tuning a GREAT AI model at https://deeplearning.ai!!!"
    ```

- Sau khi tiền xử lý nó sẽ trở thành như sau: 

![M1_W1_07_Preprocessing](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_07_Preprocessing.png)

[*tun, great, ai, model*] Do đó, bạn có thể thấy cách chúng tôi **loại bỏ các ký tự xử lý**, phân tách thành các từ, **loại bỏ các từ dừng**, thực hiện chuyển đổi gốc và **chuyển đổi mọi thứ thành chữ thường.**

#### 8. Putting it All Together
- Nhìn chung, bạn bắt đầu với một văn bản cho trước, bạn thực hiện tiền xử lý, sau đó bạn trích xuất đặc điểm để chuyển đổi văn bản thành biểu diễn số như sau:

![M1_W1_08_PuttingIAT](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_08_PuttingIAT.png)

- **X** của bạn, trở thành kích thước (*m*, 3) như sau:

![M1_W1_08_PuttingIAT_01](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_08_PuttingIAT_01.png)

- Khi triển khai bằng code, nó sẽ như sau:

![M1_W1_08_PuttingIAT_02](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_08_PuttingIAT_02.png)

- Bạn có thể thấy ở bước cuối cùng, bạn đang lưu trữ các **tính năng được trích xuất dưới dạng các hàng** (extract_features) trong ma trận **X** và bạn có *m* ví dụ này.

#### 9. Logistic Regression Overview

- Đây là cái nhìn tổng quan về **hồi quy logistic** (logistic regression).
- Bạn sẽ sử dụng các **tính năng** (features) đã trích xuất để dự đoán một **tweet** có tâm lý tích cực hay tiêu cực.
- Hồi quy logistic sử dụng một **hàm sigmoid** (sigmoid function), xuất ra một xác suất từ 0 đến 1.
> Overview of logistic regression

![M1_W1_09_Logistic Regression Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_09_Logistic_Regression_Overview.png)

- Trong **máy học được giám sát** (supervised machine learning), bạn có các tính năng đầu vào và nhãn. Bạn dùng một hàm với các tham số để ánh xạ các đối tượng với nhãn đầu ra.
- Để có ánh xạ tối ưu, bạn **giảm thiểu hàm chi phí** (cost function) bằng cách so sánh đầu ra **Y hat** với nhãn thật **Y**. Các **tham số** (parameters) được cập nhật lặp lại cho đến khi chi phí được giảm thiểu.
- Đối với hồi quy logistic, Hàm F trong hình là hàm Sigmoid
> Biểu đạt bằng phương trình

![M1_W1_10_Logistic_Regression_Overview_02](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_10_Logistic_Regression_Overview_02.png)
- Đối với hàm hồi quy logistic (H) là **hàm sigmoid**, phụ thuộc vào tham số **Theta** và vector tính năng **X dấu trên i** (quan sát thứ i, hoặc tweet thứ i).
- Hàm sigmoid tiếp cận 0 khi tích chấm của **Theta transpose X** ($\theta^T X$) tiến đến âm vô cực, và tiếp cận 1 khi nó tiến đến dương vô cực.
- Để phân loại, cần một **ngưỡng** (threshold), thường là **0.5**.
- Giá trị 0.5 tương ứng với **tích chấm** ($\theta^T X$) bằng 0.
- Khi tích chấm $\ge 0$, dự đoán là dương. Khi tích chấm $< 0$, dự đoán là âm.
> Một ví dụ được đưa ra trong bối cảnh **phân tích tình cảm** (sentiment analysis) tweet.

![M1_W1_10_Logistic_Regression_Overview_03](https://github.com/DazielNguyen/NLP301c/blob/main/Image%20on%20courses/M1_W1_10_Logistic_Regression_Overview_03)

- Sau **tiền xử lý** (preprocessing) (ví dụ: chữ thường, giảm từ về gốc như 'tun'), bạn trích xuất các tính năng thành một vector.
- Vector này bao gồm một **đơn vị thiên vị** (bias unit) và các tính năng (như tổng tần số tích cực và tiêu cực).
- Giả sử đã có bộ tham số **Theta** tối ưu, bạn có thể nhận được giá trị của **hàm sigmoid** (ví dụ 4.92 trong script) và dự đoán một tình cảm tích cực.
- Bây giờ bạn đã biết ký hiệu (notation) để **đào tạo** (train) một yếu tố trọng lượng **Theta**.
- Video tiếp theo sẽ nói về cơ chế (mechanics) đằng sau việc đào tạo bộ phân loại hồi quy logistic.

## Week 2: 
- Sử dụng phân loại Naive Bayes trên cùng một vấn đề

## Week 3: 
- Tìm hiểu về các mô hình khôn gian Vector. 
- Tim hiểu về các biểu diễn các tài liệu văn bản như tweet, artical, truy vấn hoặc bất kỳ đối tượng nào có chứa văn bản dưới dạng một Vector. Điều này quan trọng trong việc truy xuất thông tin trong tập chỉ mục, trong xếp hạng liên quan và cũng như trong lọc thông tin.

## Week 4: 
- Xây dựng hệ thống dịch máy cơ bản đầu tiền của bạn và bạn sẽ sử dụng locality sensitive hashing để cải thiện hiệu suất của Nearest Neighbor Search. 