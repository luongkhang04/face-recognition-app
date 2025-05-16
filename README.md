# Sử dụng các mô hình Deep Learning cho bài toán nhận diện khuôn mặt
## Mở đầu
Nhận diện khuôn mặt là một trong những bài toán quan trọng trong thị giác máy tính (Computer Vision), với nhiều ứng dụng thực tiễn như: kiểm soát truy cập, giám sát an ninh, xác thực danh tính, điểm danh tự động,...

Sự phát triển của các mô hình học sâu (Deep Learning), đặc biệt là các mạng tích chập (Convolutional Neural Network) như ResNet, Inception, và các kiến trúc embedding đã mở ra nhiều phương pháp hiệu quả cho bài toán này. Trong dự án này, tôi tập trung triển khai và so sánh hai phương pháp phổ biến: FaceNet (Triplet Loss) và ArcFace (Additive Angular Margin Loss).

Mục tiêu dự án:
* Triển khai và huấn luyện các mô hình embedding với hai phương pháp: FaceNet và ArcFace.
* So sánh hiệu quả của hai phương pháp trên cùng một bộ dữ liệu.
* Xây dựng phần mềm minh hoạ việc sử dụng mô hình để nhận diện khuôn mặt từ webcam.

## Phương pháp Arcface
ArcFace (Additive Angular Margin Loss) là phương pháp huấn luyện mô hình nhận diện khuôn mặt bằng cách tối ưu hóa khoảng cách góc giữa các vector đặc trưng (embedding vector) của các lớp khác nhau.

### Softmax Loss
Hàm loss phổ biến nhất cho bài toán phân loại là hàm Softmax Loss, công thức như sau:

$$L_1=-\log{\left(\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^{N}e^{W_j^Tx_i+b_j}}\right)}$$

Trong đó:
*	$x_i\in\mathbb{R}^d$ là vector đặc trưng (embedding vector) của mẫu $i$, thuộc lớp $y_i$.
*	Kích thước vector đặc trưng $d$ được đặt là 128.
*	$W_j\in R^d$ là cột thứ $j$ của trọng số $W\in R^{d\times N}$.
*	$b_j\in R$ là hệ số bias.
*	$N$ là số lượng lớp.

Softmax truyền thống được sử dụng rộng rãi trong các bài toán nhận diện khuôn mặt bằng học sâu. Tuy nhiên, hàm mất mát softmax không tối ưu rõ ràng vector đặc trưng để đảm bảo:
*	Tính tương đồng cao trong cùng một lớp.
*	Tính phân biệt tốt giữa các lớp khác nhau.

Điều này làm giảm hiệu suất trong các tình huống khó như:
*	Biến đổi dáng mặt (tư thế) lớn.
*	Khoảng cách tuổi tác lớn.
*	Các bài toán quy mô lớn.

### Thêm margin cho Softmax Loss

* Để đơn giản, ta giả sử $b_j=0$. Khi đó, logit được viết lại dưới dạng: $W_j^Tx_i=|W_j|\cdot|x_i|\cdot\cos{\theta_j}$ với $\theta_j$ là góc giữa trọng số lớp $W_j$ và đặc trưng $x_i$.
* Áp dụng chuẩn hóa $L_2$ cho cả $W_j$ và $x_i$: $|W_j|=|x_i|=1$
* Nhân $x_i$ với một hệ số scale $s$.

Bước chuẩn hóa nhằm mục đích khiến cho việc phân lớp chỉ phụ thuộc vào góc $\theta_j$. Khi đó, các vector đặc trưng phân bố trên một siêu cầu bán kính $s$:

$$L_2=-\log{\left(\frac{e^{s\cos{\theta_{y_i}}}}{e^{s\cos{\theta_{y_i}}}+\sum_{j=1,j\neq y_i}^{N}e^{s\cos{\theta_j}}}\right)}$$

Ta phạt một khoảng cách góc phụ $m<0$ vào góc $\theta_{y_i}$ giữa vector đặc trưng $x_i$ và trọng số lớp đúng $W_{y_i}$ để tăng sự tương đồng trong cùng lớp và tăng độ phân biệt giữa các lớp khác nhau.

Hàm mất mát ArcFace:

$$L_3=-\log{\left(\frac{e^{s\cos{\left(\theta_{y_i}+m\right)}}}{e^{s\cos{\left(\theta_{y_i}+m\right)}}+\sum_{j=1,j\neq y_i}^{N}e^{s\cos{\theta_j}}}\right)}$$

### Mô hình triển khai
* Backbone: ResNet-50. Thay đổi số chiều đầu ra tầng kết nối đầy đủ.
* Loss function: ArcFace
* Đầu ra là vector embedding kích thước 128 chiều.

## Phương pháp Facenet
### Triplet Loss
Mô hình embedding được biểu diễn bởi hàm $f\left(x\right)\in R^d$. Nó ánh xạ (embed) một ảnh $x$ vào không gian Euclide $d$-chiều. Ta chuẩn hóa embedding này nằm trên mặt cầu đơn vị, tức là $|f\left(x\right)|_2=1$

Trong bài toán này, ta mong muốn đảm bảo rằng một ảnh anchor $x_i^a$ của một người cụ thể sẽ gần hơn tất cả các ảnh positive $x_i^p$ (cùng người) so với bất kỳ ảnh negative $x_i^n$ nào (người khác).

Vì vậy, ta muốn đảm bảo điều kiện lề:

$$|f\left(x_i^a\right)-f\left(x_i^p\right)|_2^2+\alpha<|f\left(x_i^a\right)-f\left(x_i^n\right)|_2^2$$
$$\forall\left(f\left(x_i^a\right),f\left(x_i^p\right),f\left(x_i^n\right)\right)\in\mathcal{T}$$

Trong đó:
*	$\alpha$ là khoảng cách lề (margin) cần được đảm bảo giữa cặp positive và negative.
*	$\mathcal{T}$ là tập hợp tất cả các bộ ba (triplet) có thể trong tập huấn luyện, và có số lượng $N$.

Hàm mất mát Triplet loss:

$$L=\sum_{i=1}^{N}\left[|f\left(x_i^a\right)-f\left(x_i^p\right)|_2^2-|f\left(x_i^a\right)-f\left(x_i^n\right)|_2^2+\alpha\right]^{+}$$

Dấu $\left[\right]^{+}$ nghĩa là chỉ lấy giá trị dương hoặc 0: $\left[x\right]^{+}=\max{\left(x,0\right)}$. 
 
### Chọn bộ ba (triplet)
Việc sinh tất cả các bộ ba có thể có sẽ tạo ra rất nhiều bộ ba dễ (tức là đã thỏa mãn điều kiện lề). Những bộ ba này không đóng góp vào việc huấn luyện, nhưng vẫn phải đi qua mạng, dẫn đến hội tụ chậm. Vì vậy, ta cần chọn các bộ ba khó (hard triplets) - tức là những bộ ba vi phạm ràng buộc - để đảm bảo mạng học được tốt hơn.

Cụ thể, với một ảnh anchor $x_i^a$, ta muốn chọn ảnh positive khó $x_i^p$ sao cho:

$$x_i^p=\arg{\max_{x_i^p}{|}}f\left(x_i^a\right)-f\left(x_i^p\right)|_2^2$$

và chọn ảnh negative khó $x_i^n$ sao cho:

$$x_i^n=\arg{\min_{x_i^n}{|}}f\left(x_i^a\right)-f\left(x_i^n\right)|_2^2$$

Tuy nhiên, việc tính toán argmin và argmax trên toàn bộ tập huấn luyện là không khả thi vì:
*	Tốn tài nguyên cực lớn,
*	Dễ bị ảnh hưởng bởi những ảnh bị gán nhãn sai hoặc chất lượng kém, khiến các ví dụ khó trở nên không đáng tin cậy.

Có hai cách tiếp cận phổ biến để giải quyết vấn đề này:
*	Sinh bộ ba offline sau mỗi n bước huấn luyện: dựa trên checkpoint gần nhất của mạng, tính argmin và argmax trên một phần dữ liệu nhỏ.
*	Sinh bộ ba online: Chọn positive/negative khó trong mỗi mini-batch.

Về cách chọn ảnh negative, chọn negative khó nhất đôi khi khiến mô hình rơi vào cực trị cục bộ xấu khi bắt đầu huấn luyện, đặc biệt có thể dẫn đến sụp đổ mô hình $(f(x)\ =\ 0)$. Để tránh điều này, ta chọn negative bán khó (semi-hard negative), tức là các ảnh negative sao cho:

$$|f\left(x_i^a\right)-f\left(x_i^p\right)|_2^2<|f\left(x_i^a\right)-f\left(x_i^n\right)|_2^2<|f\left(x_i^a\right)-f\left(x_i^p\right)|_2^2+\alpha$$

Các negative bán khó nằm gần anchor, nhưng vẫn xa hơn positive, và nằm bên trong khoảng margin, vừa khó, vừa đảm bảo an toàn cho huấn luyện.

Trong dự án này, ta chọn sinh bộ ba offline sau từng bước huấn luyện. Với mỗi anchor, ta tìm chọn positive và negative trong k hàng xóm gần nhất của anchor đó.

### Sử dụng khoảng cách Manhattan
Trong khi phương pháp Arcfface tối ưu khoảng cách góc, thì phương pháp Facenet tối ưu khoảng cách Euclid. Ngoài hai phương pháp này, dự án còn triển khai Facenet với khoảng cách Manhattan.

<img src="https://github.com/user-attachments/assets/9fb4695f-fa6c-45bd-8bc2-a42541873052" alt="Khoảng cách Euclid so với khoảng cách Manhattan" width="400"/>

Giả sử các vector đặc trưng A, B, C có 2 chiều. Khoảng cách Euclid AB và AC bằng nhau, nhưng A-C khác nhau ở 2 đặc trưng trong khi A-B khác nhau chỉ ở 1 đặc trưng. Trong giả định này, sử dụng khoảng cách Manhattan giúp phân biệt A-C tốt hơn.

Khoảng cách Manhattan còn có khả năng chống nhiễu tốt hơn. Khoảng cách L₁ phạt các sai khác lớn theo kiểu tuyến tính thay vì theo bình phương như L₂. Nếu một chiều nào đó trong embedding bị nhiễu hoặc lỗi, thì độ lệch lớn ở chiều đó sẽ không chi phối toàn bộ khoảng cách như trong L₂.

Trong không gian nhiều chiều, L₁ có khả năng làm nổi bật các khác biệt nhỏ trải rộng trên nhiều chiều tốt hơn, trong khi L₂ thường bị chi phối bởi một vài sai khác lớn. Nếu mỗi chiều embedding biểu diễn một thuộc tính độc lập (ví dụ: góc nhìn, ánh sáng, đặc trưng khuôn mặt), thì L₁ sẽ cộng dồn sai khác nhỏ ở mỗi chiều.

### Mô hình triển khai
* Backbone: Inception/Resnet
* Loss function: Semi-hard Triplet Loss
* Độ đo khoảng cách: Euclid với chuẩn hóa L₂, Manhattan với chuẩn hóa L₁.
* Cách chọn triplet: Với từng anchor, chọn positive xa nhất và negative gần nhất thỏa mãn $d_{ap}<d_{an}<d_{ap}+\alpha$
* Đầu ra là vector embedding kích thước 128 chiều.

## Thực nghiệm

## Phần mềm
