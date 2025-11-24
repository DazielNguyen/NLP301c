# **Module 04 - Natural Language Processing with Attention Models Models**
## **Week 3: Question Answering**
---
### **Week 3 Overview**
---
**Transfer Learning**

* `Transfer learning` cho phép các `models` tận dụng kiến thức thu được từ `task` này để cải thiện `performance` trên `task` khác, giảm thời gian `training` và yêu cầu về dữ liệu.
* Một ví dụ là sử dụng một `pre-trained model` trên các đánh giá phim để dự đoán `ratings` cho các đánh giá khóa học, bắt đầu với các `weights` hiện có thay vì khởi tạo từ đầu.

**Question Answering**

* Khóa học đề cập đến hai loại `question answering`: `context-based` (dựa trên ngữ cảnh), sử dụng `context` được cung cấp để tìm câu trả lời, và `closed book` (sách đóng), tạo ra câu trả lời mà không cần `context`.
* Các `models` như `BERT` sử dụng `bi-directional context` để nâng cao `performance`, trong khi `T5` có thể xử lý nhiều `tasks`, chẳng hạn như trả lời câu hỏi và dự đoán `ratings`, bằng cách sử dụng một `model` duy nhất.

**Key Takeaways**

* Các đổi mới trong các phương pháp `training`, chẳng hạn như `transfer learning`, có thể cải thiện đáng kể `model performance`.
* Càng có nhiều dữ liệu sẵn có để `training`, `model performance` càng có xu hướng tốt hơn, như thấy với `T5 model` được `trained` trên một `large dataset`.

> Chào mừng đến với Tuần 3! Trong tuần này, bạn sẽ tìm hiểu về `transfer learning` và cụ thể bạn sẽ hiểu cách `T5` và `BERT` thực sự hoạt động.

![01_Week_3_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/01_Week_3_Overview.png)

> Trong hình ảnh phía trên, bạn có thể thấy cách một `model` ban đầu được `trained` trên một loại `sentiment classification`, giờ đây có thể được sử dụng cho `question answering`.

> Một `model state of the art` khác sử dụng `multi tasking`. Ví dụ, cùng một `model` có thể được sử dụng cho `sentiment analysis`, `question answering`, và nhiều thứ khác.

![02_Week_3_Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/02_Week_3_Overview.png)

> Các loại `models` mới này sử dụng rất nhiều dữ liệu. Ví dụ, `C4` (`colossal cleaned crawled corpus`) có dung lượng khoảng 800 GB trong khi toàn bộ `Wikipedia` tiếng Anh chỉ là 13 GB!


---
### **Transfer Learning in NLP**
---

Nội dung bài giảng này tập trung vào khái niệm `transfer learning` trong `natural language processing` (`NLP`) và ứng dụng của nó trong các `models` khác nhau.

**Transfer Learning Overview**

* `Transfer learning` liên quan đến việc sử dụng các `pre-trained models` để cải thiện `performance` trên các `tasks` cụ thể.
* Nó có thể được triển khai thông qua `feature-based learning` (ví dụ: `word vectors`) hoặc `fine-tuning` các `models` hiện có.


**Feature-Based Learning vs. Fine-Tuning**

* `Feature-based learning` sử dụng các `embeddings` làm `input features` cho các `models` khác nhau để đưa ra `predictions`.
* `Fine-tuning` liên quan đến việc điều chỉnh các `weights` của một `pre-trained model` cho một `downstream task` cụ thể, chẳng hạn như `sentiment analysis`.

**Data and Performance**

* Số lượng và chất lượng của dữ liệu ảnh hưởng đáng kể đến `model performance`; nhiều dữ liệu hơn thường dẫn đến kết quả tốt hơn.
* `Labeled data` thường khan hiếm so với `unlabeled data`, cái mà có thể được tận dụng trong các `self-supervised tasks` để tạo `input features` và `targets`.

**Pre-Training and Downstream Tasks**

* Các `pre-training tasks` có thể bao gồm `language modeling`, nơi các `models` dự đoán các `masked words` hoặc câu tiếp theo.
* `Fine-tuning` có thể được áp dụng cho các `tasks` khác nhau, bao gồm dịch thuật (`translation`), tóm tắt (`summarization`), và `question answering`, sử dụng `pre-trained model`.

> Có ba lợi ích chính của `transfer learning`:

* Giảm `training time` (thời gian huấn luyện)
* Cải thiện `predictions` (dự đoán)
* Cho phép bạn sử dụng các `datasets` nhỏ hơn

> Hai phương pháp mà bạn có thể sử dụng cho `transfer learning` là:

![03_Transfer_Learning_in_NLP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/03_Transfer_Learning_in_NLP.png)

> Trong `feature based transfer learning`, bạn học `word embeddings` bằng cách `training` một `model` và sau đó bạn sử dụng các `word embeddings` đó trong một `model` khác cho một `task` khác.

> Khi `fine tuning`, bạn có thể sử dụng chính xác cùng một `model` và chỉ chạy nó trên một `task` khác. Đôi khi khi `fine tuning`, bạn có thể giữ các `model weights` cố định và chỉ thêm một `layer` mới mà bạn sẽ `train`. Những lúc khác bạn có thể từ từ `unfreeze` (rã đông) các `layers` từng cái một. Bạn cũng có thể sử dụng `unlabelled data` khi `pre-training`, bằng cách `masking` các từ và cố gắng `predict` từ nào đã bị `mask`.

![04_Transfer_Learning_in_NLP](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/04_Transfer_Learning_in_NLP.png)

> Ví dụ, trong hình vẽ trên, chúng ta cố gắng `predict` từ "friend". Điều này cho phép `model` của bạn nắm bắt được cấu trúc tổng thể của dữ liệu và giúp `model` học được một số mối quan hệ trong các từ của một `sentence`.

---
### **ELMo, GPT, BERT, T5**
---

Nội dung thảo luận về sự tiến hóa của các `models natural language processing` (`NLP`) khác nhau cùng các ưu điểm và nhược điểm tương ứng của chúng.

**Model Evolution**

* Dòng thời gian bao gồm các `models` như `Continuous Bag of Words`, `ELMo`, `GPT`, `BERT`, và `T5`, làm nổi bật sự phát triển của chúng và các vấn đề mà chúng giải quyết.
* Mỗi `model` xây dựng dựa trên các khái niệm trước đó, dẫn đến những cải tiến trong việc hiểu `context` và `word embeddings`.

**Contextual Understanding**

* Tầm quan trọng của `context` trong việc hiểu các từ được nhấn mạnh, với các phương pháp như `fixed window sizes` và `bi-directional LSTMs` được khám phá.
* Cách tiếp cận `bi-directional` của `BERT` cho phép sử dụng `context` tốt hơn so với các `models` trước đó.

**Model Architectures**

* Các `architectures` khác nhau được thảo luận: `GPT` sử dụng `decoder-only stack`, trong khi `BERT` sử dụng `encoder-only stack`.
* `T5` kết hợp cả `encoder` và `decoder stacks`, cho thấy `performance` được cải thiện trong các `tasks` khác nhau thông qua các `multi-task training strategies`.

Bản tóm tắt này gói gọn các điểm chính liên quan đến sự phát triển và chức năng của các `NLP models`.

> Các `models` được đề cập trong video trước đã được khám phá theo thứ tự sau:

![05_ELMo_GPT_BERT_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/05_ELMo_GPT_BERT_T5.png)


> Trong `CBOW`, bạn muốn mã hóa một từ dưới dạng một `vector`. Để làm điều này, chúng ta sử dụng `context` trước từ và `context` sau từ và chúng ta sử dụng `model` đó để học và tạo ra các `features` cho từ. Tuy nhiên, `CBOW` sử dụng một `fixed window C` (cho `context`).

> `ElMo` sử dụng một `bi-directional LSTM`, đây là một phiên bản khác của `RNN` và bạn có các `inputs` từ bên trái và bên phải.

> Sau đó `Open AI` đã giới thiệu `GPT`, một `uni-directional model` sử dụng `transformers`. Mặc dù `ElMo` là `bi-directional`, nó vẫn gặp một số vấn đề như nắm bắt các `longer-term dependencies`, điều mà `transformers` giải quyết tốt hơn nhiều.

> Sau đó, `Bi-directional Encoder Representation from Transformers` (`BERT`) được giới thiệu, cái mà tận dụng các `bi-directional transformers` như tên gọi của nó.

> Cuối cùng, `T5` được giới thiệu, cái mà sử dụng `transfer learning` và sử dụng cùng một `model` để `predict` (dự đoán) trên nhiều `tasks`. Dưới đây là một minh họa về cách nó hoạt động:

![06_ELMo_GPT_BERT_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/06_ELMo_GPT_BERT_T5.png)

---
### **Bidirectional Encoder Representations from Transformers (BERT)**
---

Nội dung tập trung vào `Bidirectional Encoder Representations from Transformers` (`BERT`), một `model` sử dụng `transformer architecture` để xử lý các `inputs` theo cả hai hướng (`bidirectional`).

**BERT Architecture**

* `BERT` sử dụng một `multi-layer bidirectional transformer` với `positional embeddings`.
* `Base model` bao gồm 12 `layers`, 12 `attention heads`, và 110 triệu `parameters`.

**Pre-training Process**

* `BERT` được `pre-trained` trên dữ liệu không có nhãn (`unlabeled data`) bằng cách sử dụng các `tasks` như `masked language modeling` và `next sentence prediction`.
* Trong quá trình `pre-training`, 15% số từ trong `input` được `masked` (che), và `model` dự đoán các từ bị `masked` này.

**Fine-tuning**

* Sau `pre-training`, `BERT` được `fine-tuned` trên dữ liệu có nhãn (`labeled data`) cho các `downstream tasks` cụ thể.
* Các `parameters` của `model` được điều chỉnh để cải thiện `performance` trên các `tasks` này, chẳng hạn như `classification` hoặc `prediction`.

> Bây giờ bạn sẽ tìm hiểu về `BERT architecture` và hiểu cách `pre-training` hoạt động.

![07_BERT](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/07_BERT.png)


> Có hai bước trong `BERT framework`: `pre-training` và `fine-tuning`. Trong quá trình `pre-training`, `model` được `trained` trên dữ liệu không có nhãn (`unlabeled data`) thông qua các `pre-training tasks` khác nhau. Đối với `fine-tuning`, `BERT model` đầu tiên được `initialize` (khởi tạo) với các `pre-trained parameters`, và tất cả các `parameters` đều được `fine-tuned` bằng cách sử dụng dữ liệu có nhãn (`labeled data`) từ các `downstream tasks`.

> Ví dụ, trong hình ảnh phía trên, bạn nhận được các `embeddings` tương ứng cho các từ `input`, bạn chạy nó qua một vài `transformer blocks`, và sau đó bạn đưa ra `prediction` tại mỗi `time point` $T_i$.

> Trong quá trình `pre-training`:
* Chọn 15% `tokens` một cách ngẫu nhiên: `mask` chúng 80% thời gian, thay thế chúng bằng một `random token` 10% thời gian, hoặc giữ nguyên 10% thời gian.
* Có thể có nhiều `masked spans` (khoảng che) trong một câu.
* `Next sentence prediction` cũng được sử dụng khi `pre-training`.

---
### **BERT Objective**
---

Nội dung tập trung vào `input representation` và `objective` của `BERT model` trong `natural language processing`.

**Input Representation**

* `BERT` sử dụng `position embeddings` để chỉ ra vị trí của mỗi từ trong một câu.
* `Segment embeddings` phân biệt giữa `sentence A` và `sentence B`, điều này cần thiết cho `next sentence prediction`.

**Combining Inputs**

* `Token embeddings`, bao gồm một **CLS token** cho điểm bắt đầu và một **SEP token** cho điểm kết thúc của câu, được cộng tổng với `position` và `segment embeddings` để tạo ra `final input`.
* `Input` được chuyển đổi thành các `embeddings` được xử lý qua các `transformer blocks`.

**BERT Objective**

* `Model` sử dụng một **Multi-Mask language model** dùng `cross-entropy loss` để dự đoán các từ bị `masked`.
* Một **binary loss** được thêm vào cho **next sentence prediction**, xác định xem hai câu có theo sau nhau hay không.

> Chúng ta sẽ bắt đầu bằng cách trực quan hóa (`visualizing`) `input`.

![08_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/08_BERT_Objective.png)

> Các `input embeddings` là tổng của các `token embeddings`, các `segmentation embeddings` và các `position embeddings`.
* Các `input embeddings`: bạn có một **CLS token** để chỉ ra điểm bắt đầu của câu và một **SEP** để chỉ ra điểm kết thúc của câu.
* Các `segment embeddings`: cho phép bạn chỉ ra đó là `sentence A` hay `B`.
* Các `positional embeddings`: cho phép bạn chỉ ra vị trí của từ trong câu.

![09_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/09_BERT_Objective.png)


> `C token` trong hình ảnh phía trên có thể được sử dụng cho các mục đích phân loại (`classification purposes`). Cặp `unlabeled sentence A/B` sẽ phụ thuộc vào những gì bạn đang cố gắng `predict` (dự đoán), nó có thể bao gồm từ `question answering` đến `sentiment` (trong trường hợp đó, câu thứ hai có thể chỉ là trống). `BERT objective` được định nghĩa như sau:

![10_BERT_Objective](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/10_BERT_Objective.png)

> Bạn chỉ cần kết hợp các `losses`!

---
### **Fine tuning BERT**
---

Nội dung này tập trung vào việc `fine-tuning` `BERT model` cho các `natural language processing tasks` khác nhau.

**Fine-Tuning BERT for Different Tasks**

* `BERT` có thể được `fine-tuned` cho các `tasks` như `MNLI` (`Multi-Genre Natural Language Inference`) bằng cách sử dụng một `hypothesis` và `premise` thay vì các `sentence pairs`.
* Đối với `Named Entity Recognition` (`NER`), `model` nhận một câu và các `tags` tương ứng để xác định các `entities`.

**Visual Representation of Input**

* `Input structure` cho `question answering` bao gồm một câu hỏi và một đoạn văn (`paragraph`) mà từ đó câu trả lời được suy ra.
* Đối với `NER`, `input` bao gồm một câu được ghép nối với các `named entities` của nó, trong khi `MNLI` sử dụng định dạng `hypothesis` và `premise`.

**Summary of Input Types**

* Các `tasks` khác nhau yêu cầu các định dạng `input` cụ thể: các `classification tasks` sử dụng văn bản với `labels`, `question answering` sử dụng các cặp câu hỏi-đoạn văn, và `summarization` có thể liên quan đến các bài báo và bản tóm tắt của chúng.
* Việc hiểu các `input structures` này là rất quan trọng để `fine-tuning` `BERT` hiệu quả cho các ứng dụng khác nhau.

> Khi bạn đã có một `pre-trained model`, bạn có thể `fine tune` nó trên các `tasks` khác nhau.

![11_Fine_tuning_BERT](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/11_Fine_tuning_BERT.png)

> Ví dụ, cho một `hypothesis`, bạn có thể xác định `premise`. Cho một câu hỏi, bạn có thể tìm thấy câu trả lời. Bạn cũng có thể sử dụng nó cho `named entity recognition`, `paraphrasing sentences`, `sequence tagging`, `classification` và nhiều `tasks` khác.

![12_Fine_tuning_BERT](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/12_Fine_tuning_BERT.png)

---
### **Transformer: T5**
---

Nội dung tập trung vào `T5 model`, cái mà được tận dụng cho các `natural language processing` (`NLP tasks`) khác nhau và áp dụng một chiến lược `training` tương tự như `BERT`.

**T5 Model Overview**

* `T5`, hay **Text-to-Text Transfer Transformer**, có thể được áp dụng cho các `tasks` như `classification`, `question answering`, `machine translation`, `summarization`, và `sentiment analysis`.
* `Model architecture` bao gồm một `encoder-decoder structure` với các `transformers`, sử dụng `transfer learning` và `masked language modeling` để `training`.

**Training Process**

* `Training` liên quan đến việc `masking` các từ nhất định trong một văn bản và thay thế chúng bằng các `tokens`, cái mà `model` học cách dự đoán.
* `Architecture` có các **fully visible attention** trong `encoder` và **causal attention** trong `decoder`, với các chiến lược `masking` khác nhau được áp dụng.

**Attention Mechanisms**

* Bài giảng thảo luận về các `attention types` khác nhau, bao gồm `prefix language model attention` và `causal masking`.
* `T5 model` bao gồm 12 `transformer blocks` và có xấp xỉ 220 triệu `parameters`, cho phép nó xử lý nhiều `NLP tasks` một cách hiệu quả.

> Một trong những kỹ thuật chính cho phép `T5 model` đạt tới `state of the art` là khái niệm `masking`: 

![13_Transformer_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/13_Transformer_T5.png)

> Ví dụ, bạn biểu diễn "for inviting" bằng `<X>` và "last" bằng `<Y>`, sau đó `model` dự đoán `<X>` nên là gì và `<Y>` nên là gì. Đây chính xác là những gì chúng ta đã thấy trong `BERT loss`. Bạn cũng có thể `mask out` một vài vị trí, không chỉ một. `Loss` chỉ được tính trên `mask` đối với `BERT`, còn đối với `T5` thì nó được tính trên `target`.

![14_Transformer_T5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/14_Transformer_T5.png)

> Vì vậy, chúng ta bắt đầu với biểu diễn `encoder-decoder` cơ bản. Ở đó bạn có một **fully visible attention** trong `encoder` và sau đó là **causal attention** trong `decoder`. Vì vậy, các đường màu xám nhạt tương ứng với `causal masking`. Và các đường màu xám đậm tương ứng với `fully visible masking`.

> Ở giữa, chúng ta có `language model` bao gồm một `single transformer layer stack`. Và nó được nạp đầu vào (`fed`) bằng cách `concatenation` (nối) của các `inputs` và `target`. Vì vậy, nó sử dụng `causal masking` xuyên suốt như bạn thấy vì chúng đều là các đường màu xám. Và bạn có $X_1$ đi vào bên trong, bạn nhận được $X_2$, $X_2$ đi vào `model` và bạn nhận được $X_3$ và cứ thế tiếp diễn.

> Ở bên phải, chúng ta có `prefix language model` tương ứng với việc cho phép `fully visible masking` trên các `inputs` như bạn thấy với các mũi tên màu tối. Và sau đó là `causal masking` ở phần còn lại.

---
### **Multi-Task Training Strategy**
---

Nội dung này tập trung vào việc `training` một `model` để thực hiện các `Natural Language Processing` (`NLP tasks`) khác nhau bằng cách sử dụng một `multitask training strategy`.

**Multitask Training Strategy**

* Một `model` có thể được `trained` để thực hiện nhiều `tasks` bằng cách thêm các `tags` cụ thể, chẳng hạn như "translate" cho các `translation tasks` hoặc "summarize" cho các `summarization tasks`.
* Đối với các `tasks` như `entailment prediction`, `model` được cung cấp `premises` và `hypotheses` để phân loại các mối quan hệ.

**Data Training Strategies**

* Các ví dụ về chiến lược `data training` bao gồm `proportional mixing` (trộn tỷ lệ), nơi các tỷ lệ dữ liệu bằng nhau từ mỗi `task` được sử dụng, và `equal mixing` (trộn đều), nơi các mẫu bằng nhau được lấy bất kể kích thước `dataset`.
* `Temperature-scaled mixing` là một cách tiếp cận lai điều chỉnh các `parameters` cho một sự pha trộn cân bằng.

**Gradual Unfreezing and Adapter Layers**

* **Gradual unfreezing** (rã đông dần dần) liên quan đến việc `unfreezing` một `layer` của `neural network` tại một thời điểm để `fine-tuning`.
* **Adapter layers** là các `neural networks` bổ sung được thêm vào mỗi `transformer block`, cho phép `fine-tuning` mà không làm thay đổi `model's structure`.

**Evaluation with GLUE Benchmark**

* **GLUE benchmark** được giới thiệu như một phương pháp để đánh giá `performance` của `model` trên các `NLP tasks` khác nhau, với trọng tâm là đạt được kết quả `state-of-the-art`.

> Đây là lời nhắc về cách `T5 model` hoạt động:

![15_Multi-Task_Training_Strategy](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/15_Multi-Task_Training_Strategy.png)

> Bạn có thể thấy rằng bạn chỉ cần thêm một `prefix` nhỏ vào `input` và `model` sẽ giải quyết `task` cho bạn. Có nhiều `tasks` mà `T5 model` có thể thực hiện cho bạn.

> Có thể xây dựng hầu hết các `NLP tasks` dưới định dạng “`text-to-text`” – tức là một `task` nơi `model` được cung cấp một số văn bản để làm `context` hoặc điều kiện hóa (`conditioning`) và sau đó được yêu cầu tạo ra một số văn bản đầu ra. `Framework` này cung cấp một `objective training` nhất quán cho cả `pre-training` và `fine-tuning`. Cụ thể, `model` được `trained` với `maximum likelihood objective` (sử dụng “`teacher forcing`”) bất kể đó là `task` gì.

> **Training data strategies**

* **Examples-proportional mixing** (Trộn tỷ lệ theo ví dụ): lấy mẫu theo tỷ lệ kích thước `dataset` của mỗi `task`.
* **Temperature scaled mixing** (Trộn theo tỷ lệ nhiệt độ): điều chỉnh “`temperature`” (nhiệt độ) của `mixing rates`. `Parameter temperature` này cho phép bạn cân nhắc một số ví dụ hơn những ví dụ khác. Để triển khai `temperature scaling` với `temperature` $T$, chúng ta nâng `mixing rate` $r_m$ của mỗi `task` lên lũy thừa $1/T$ và `renormalize` các `rates` sao cho tổng của chúng bằng 1. Khi $T = 1$, cách tiếp cận này tương đương với `examples-proportional mixing` và khi $T$ tăng, các tỷ lệ trở nên gần hơn với `equal mixing`.
* **Equal mixing** (Trộn đều): Trong trường hợp này, bạn lấy mẫu các ví dụ từ mỗi `task` với xác suất bằng nhau. Cụ thể, mỗi ví dụ trong mỗi `batch` được lấy mẫu ngẫu nhiên đồng nhất từ một trong các `datasets` bạn `train` trên.

> **Fine tuning example**

![16_Multi-Task_Training_Strategy](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/16_Multi-Task_Training_Strategy.png)

> Bạn có thể thấy ở trên cách `fine tuning` trên một `task` cụ thể có thể hoạt động ngay cả khi bạn `pre-training` trên các `tasks` khác nhau.

---
### **GLUE Benchmark**
---

Nội dung tập trung vào `GLUE Benchmark`, một công cụ được sử dụng rộng rãi trong `natural language processing` (`NLP`) để `training`, đánh giá (`evaluating`), và phân tích các hệ thống hiểu ngôn ngữ.

**GLUE Benchmark Overview**

* `GLUE` là viết tắt của **General Language Understanding Evaluation** và bao gồm một bộ sưu tập các `datasets` cho các `NLP tasks` khác nhau.
* Nó bao gồm các `datasets` thuộc nhiều thể loại, kích thước và độ khó khác nhau, bao gồm các `tasks` như `co-reference resolution`, `sentiment analysis`, và `question answering`.

**Evaluation and Research**

* `Benchmark` này có một `leaderboard` (bảng xếp hạng) để so sánh `model performance` trên các `datasets` khác nhau.
* Các `tasks` được đánh giá bao gồm tính đúng ngữ pháp (`grammaticality`) của các câu, `sentiment analysis`, `paraphrasing` (diễn giải lại), và xác định các mâu thuẫn (`contradictions`) hoặc `entailments` (suy luận logic).

**Model Agnosticism and Transfer Learning**

* `GLUE` là **model agnostic** (bất khả tri về mô hình), cho phép bất kỳ `model` nào cũng được đánh giá trên các `datasets` của nó.
* Nó hỗ trợ `transfer learning`, cho phép các `models` học từ nhiều `datasets` để cải thiện `performance` trên các `tasks` mới trong `GLUE`.

> `General Language Understanding Evaluation` (`GLUE`) chứa:

* Một bộ sưu tập được sử dụng để `train`, đánh giá (`evaluate`), phân tích các hệ thống hiểu ngôn ngữ tự nhiên (`natural language understanding systems`).
* Các `datasets` với nhiều thể loại khác nhau, và có các kích thước và độ khó khác nhau.
* `Leaderboard` (bảng xếp hạng).

![17_GLUE_Benchmark](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2004/Image_Module_04/M4_W3/17_GLUE_Benchmark.png)

> `GLUE benchmark` này được sử dụng cho các mục đích nghiên cứu, nó là `model agnostic` (bất khả tri về mô hình), và dựa vào các `models` sử dụng `transfer learning`.

---
### **Hugging Face Introduction**
---

Nội dung tập trung vào các công cụ và tài nguyên do `Hugging Face` cung cấp để tạo điều kiện thuận lợi cho `machine learning`.

**Hugging Face Tools**

* **Transformers library** hỗ trợ nhiều `transformer architectures` khác nhau cho các dự án `natural language processing`, `computer vision`, và `speech`.
* **Datasets library** cho phép người dùng dễ dàng tải xuống và `preprocess` hơn 1.000 `datasets` khác nhau.

**Hugging Face Hub**

* **Model hub** lưu trữ hơn 15.000 `models` do cộng đồng đóng góp, cho phép người dùng chọn `models` dựa trên các `tasks` và `datasets` cụ thể.
* **Dataset hub** cung cấp hàng nghìn `datasets` với các `data set cards` toàn diện, mô tả chi tiết thiết kế và các cân nhắc khi sử dụng.

**Upcoming Labs**

* Khóa học sẽ bao gồm các `labs` thực hành, nơi người học sẽ sử dụng các công cụ của `Hugging Face` để tìm và `fine-tune` các `pre-trained models`.
* Những người tham gia sẽ có kinh nghiệm thực hành với các công cụ như `transformers` và `datasets` trong bối cảnh các ứng dụng `machine learning`.

---
### **Hugging Face I**
---

Nội dung giới thiệu `Hugging Face` và `transformers library` của nó, tập trung vào các ứng dụng trong `natural language processing` (`NLP`).

**Hugging Face Overview**

* `Hugging Face` cung cấp một hệ sinh thái được ghi chép tốt (`well-documented ecosystem`) cho `NLP`, bao gồm một khóa học để khám phá sâu hơn.
* `Transformers library` cho phép tích hợp với các `frameworks` phổ biến như `PyTorch`, `TensorFlow`, và `Flax`.

**Transformers Library Features**

* `Library` này hỗ trợ hai chức năng chính: áp dụng các `state-of-the-art transformer models` cho các `NLP tasks` khác nhau và `fine-tuning` các `pre-trained models` với các `datasets` tùy chỉnh.
* `Pipelines` đơn giản hóa quy trình bằng cách xử lý `input pre-processing`, thực thi `model`, và `output post-processing`.

**Fine-Tuning and Model Checkpoints**

* `Hugging Face` cung cấp hơn 15.000 `pre-trained model checkpoints` để `fine-tuning` các `transformer architectures` phổ biến.
* `Library` bao gồm các `tokenizers` cho `data pre-processing` và các công cụ để `training models` sử dụng `PyTorch` hoặc `TensorFlow`, cùng với các `evaluation metrics` cho `model performance`.

---
### **Hugging Face II**
---

Nội dung tập trung vào việc sử dụng `transformers library` cho các `natural language processing` (`NLP tasks`) khác nhau thông qua `pipeline object` của nó.

**Pipeline Overview**

* **Pipeline object** đơn giản hóa quy trình áp dụng các `transformer models` cho các `NLP tasks` khác nhau bằng cách xử lý `input pre-processing`, thực thi `model`, và `output post-processing`.
* Người dùng có thể chỉ định `task` cho `pipeline`, chẳng hạn như `question answering` hoặc `sentiment analysis`, và cung cấp các `inputs` cần thiết.

**Supported Tasks**

* `Library` hỗ trợ một loạt các `NLP tasks`, bao gồm `sentiment analysis`, `question answering`, và `text completion` (`fill-mask`).
* Mỗi `task` yêu cầu các `inputs` cụ thể, chẳng hạn như `context` và câu hỏi cho `question answering`, hoặc các câu có chỗ trống cho các `fill-mask tasks`.

**Model Selection**

* Người dùng có thể chọn các `model checkpoints` cụ thể cho các `pipelines` của họ, với `Hugging Face` cung cấp nhiều loại `pre-trained models` được điều chỉnh cho các `tasks` khác nhau.
* Điều quan trọng là phải chọn `checkpoint` thích hợp dựa trên các yêu cầu của `task`, vì không phải tất cả các `models` đều phù hợp cho mọi `task`.

**Model Hub**

* `Hugging Face` cung cấp một **model hub** nơi người dùng có thể tìm và lọc các `pre-trained models` dựa trên các `tasks` hoặc `datasets` mong muốn của họ.
* Giao diện `model card` cung cấp thông tin chi tiết về mỗi `model`, bao gồm mô tả và các `code snippets` để triển khai.

---
### **Hugging Face III**
---

Nội dung tập trung vào việc tận dụng các công cụ của `Hugging Face` để `fine-tuning` các `transformer models` trong `natural language processing`.

**Hugging Face Tools Overview**

* Cung cấp hơn 1.000 `datasets` cho các `tasks` cụ thể, dễ dàng truy cập thông qua `datasets library`.
* Cung cấp hơn 15.000 `model checkpoints` có thể được tải từ `transformers library`.

**Data Preparation and Tokenization**

* **Tokenizers** có sẵn để `pre-process` dữ liệu trước khi `training` và `post-process` `outputs` sau khi thực thi `model`.
* **Datasets library** được tối ưu hóa cho các `large datasets`, đơn giản hóa việc tải và `pre-processing` dữ liệu.

**Training and Evaluation**

* **Trainer object** trong `Hugging Face` cho phép `training model` dễ dàng với mã hóa tối thiểu.
* Các `metrics` được định nghĩa trước có sẵn để đánh giá `model performance`, và các `custom metrics` cũng có thể được định nghĩa.

---
### **Andrew Ng with Quoc Le**
---

**Quoc Le's Journey in AI**

* `Quoc` bắt đầu sự quan tâm của mình đối với `AI` trong thời gian trung học, tạo ra các chương trình đơn giản như một `rule-based chatbot`.
* Ông theo học bằng cử nhân tại `Australia`, nơi ông thực tập với `Alex Smola`, dẫn đến niềm đam mê của ông với `machine learning`.
* Hành trình học thuật của ông bao gồm bằng `PhD` tại `Stanford`, nơi ông chịu ảnh hưởng bởi công trình của `Andrew Ng` về `machine learning` cho `AI`.

**Contributions to NLP**

* `Quoc` đóng vai trò chủ chốt trong `Google Cat Project`, cái mà đã chứng minh khả năng của `unsupervised learning` (học không giám sát) bằng cách nhận diện mèo trong các video `YouTube`.
* Ông đồng phát triển `sequence-to-sequence model` (`seq2seq`), cái mà đã cách mạng hóa `machine translation` (dịch máy) bằng cách cho phép dịch cấp độ câu thay vì từng từ.

**Future of NLP**

* `Quoc` bày tỏ sự hào hứng về các `generative models` trong `NLP`, nhấn mạnh tiềm năng tạo ra văn bản mạch lạc và liên quan đến ngữ cảnh.
* Ông nhấn mạnh tầm quan trọng của việc cải thiện độ chính xác thực tế (`factual accuracy`) và hiểu biết thông thường (`common sense understanding`) trong `AI-generated content`.

Cuộc phỏng vấn giới thiệu những đóng góp đáng kể của `Quoc Le` cho `NLP` và những hiểu biết sâu sắc của ông về tương lai của công nghệ `AI`.
