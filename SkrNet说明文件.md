
## SkyNet网络结构

网络结构定义

- input layer : conv0
- conv layer
- pooling layer
- reorg layer
- concatenate layer

只表示相应层的信息，不代表顺序，reorg 其实是在 concat 之前才去做的

<img src="pics/image-20210608163534497.png" alt="image-20210608163534497" style="zoom:50%;" />

所包含的信息

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608164012945.png" alt="image-20210608164012945" style="zoom:50%;" />



## SkyNet 量化方案

在此量化方案中完全可以做每通道的量化，当然这需要以输出通道为划分依据，分别计算weight、bias、activation 这三个量关于每个通道的缩放因子 S 和相应的量化值 Q

<img src="pics/image-20210528115713310.png" alt="image-20210528115713310" style="zoom:30%;" />
$$
Q_BM = BM/S_B=BS_AS_W/S_B
$$
其中，激活值和权重的伸缩系数由原数据范围和量化比特数计算得到，而偏移的伸缩系数由另两个伸缩系数所确定，
$$
S_B = S_AS_W
$$
这样
$$
Q_AQ_W 和Q_B在同一尺度可以相加
$$

---

### 量化参数的获取

以 Pytorch 为工具，

一个神经网络，比如 SkyNet，以 .pth 的形式对训练好的网络进行记录

量化时，

- 创建一个 SkyNet 的网络

- 载入训练好的 .pth 数据

- 做 DFQ
  - 将所有的 Conv 换成 Qconv（在分类任务中，还需要把 Linear 层换成 Qlinear 层）
  
  - 设置 activation 量化位宽
  
  - 进行层融合，将 conv 层与其后的 BN 层融合
    - 融合之后的效果， conv 层有了 bias 参数
    - BN 层变成了 identity_layer，即输入等于输出的 “同等层”
    
  - 做权重均衡
  
  - 做偏移修复
  
  - 对 Qconv（或者是 Qlinear）层的参数做量化
    - 在这里可以得到并输出所有关于权重量化的信息
    - 根据 weight_min weight_max 和 weight 所对应的比特数可以确定其对应的缩放因子 Sw
    
  - 根据前面 BN 层数据分布的统计信息对激活值的范围做预测，得到 running_min/ running_max 值
    - 在这里可以得到并输出所有关于激活值量化的信息
    - 根据 running_min running_max 和 activation 所对应的比特数可以确定其对应的缩放因子 Sa
    
    在具体用代码实现时，采用对称量化
    
    > 关于 Qconv 的参数量化可以先不做，在设置完 running_min 和 running_max 之后加入以下代码
    >
    > - 遍历整个网络，选出其中的 Qconv 层，根据激活值的量化比特数和 running_min 和 running_max 对每一层计算其对应的 Sa，记录之 uint8。输入层范围为 (0,1) 通过 uint8 量化后范围在 (0,255)
    > - 对其中的 Qconv 层，根据 weight 的 min/max 值和 weight 的量化比特位数计算每一个通道量化后的权重值 Q_weight 和其对应的 Sw，记录之 int6
    > - 根据以上两个记录，得到 bias 对应的 Sb = Sa x Sw，据此得到量化后的 bias 值 Q_bias，记录之 int16
    > - 对本层输出进行 Requantize，得到下一层的 Q_a，即乘 Sm = Sb / Sa'，将 Sm 记录为 Qm>>17 位的形式，即 Qm = Sm x 2^17，记录之 int16
    
    存在的问题是需要写与之对应的量化推理过程，并记录中间结果部分的代码
  
- 量化推断（fake quantization 以稀疏的 32 位浮点数模拟量化后的推断效果）
  
  - 在每一层的计算中，首先根据 min/max 值对输入的激活值做量化，之后再与量化后的权重偏移参数做计算

---

量化比特位数

**#define** na 8

**#define** nw 6

**#define** nb 16

**#define** nm 17

**#define** qm 131072.0

量化数据类型

**typedef** ap_uint<8> ADT;

**typedef** ap_int<19> RDT;

**typedef** ap_int<16> BDT;

**typedef** ap_int<6> WDT;

**typedef** ap_int<16> MDT;

---

对于 conv1x1

```c
void PWCONV1X1(ADT IFM[32][43][83], RDT OFM[32][43][83], WDT WBUF1x1[32][32])
{
}
```

MAC16

​	16个14位数之和用18位数表示

通过使用 20 位的 res 和截断操作，使得计算结果总是 19 位，不会发生溢出情况

<img src="pics/image-20210601114752388.png" alt="image-20210601114752388" style="zoom:50%;" />

对于 conv3x3

```c++
void DWCONV3X3(const ADT IFM[32][43][83], RDT OFM[32][43][83], const WDT WBUF3x3[32][3][3])
{
}
```

MAC9

​	9个14位数之和用18位数表示

结果为19位数据，所以代码中的截断操作是没有必要的(一个18位数据永远不可能越19位数据的界)

<img src="pics/image-20210601150104083.png" alt="image-20210601150104083" style="zoom:50%;" />

对于 ACTIVATION 操作

```c++
void ACTIVATION(RDT IFM[32][43][83], ADT OFM[32][43][83], BDT BBUF[32], MDT MBUF[32])//MDT: multiplier for each channel = SwSa >> nM
{
}
```

用一个20位的数据表示19位的中间结果和16位的偏移之和，进行ReLU操作后，通过乘一个因子并加上截断操作恢复到激活值的级别uint8，作为下一层的输入进行后续计算

<img src="pics/image-20210601150833465.png" alt="image-20210601150833465" style="zoom:50%;" />

---

采用对称量化方案

考虑相邻的两层，对层一
$$
A*W = Q_A*S_A*Q_w*S_W=S_AS_W*Q_AQ_W=S_M*Q_AQ_W
$$
对层一对偏置，有了以上两个放缩系数可得层一偏移的放缩系数
$$
B = S_B*Q_B=S_M*Q_B
$$
层一的输出为
$$
A*W+B = S_M(Q_AQ_W+Q_B)
$$
经过 ReLU 激活函数后，其输出作为下一层的输入
$$
ReLU(A*W+B)
$$
下一层对输入进行量化，需要除以该层的放缩系数
$$
A_2=ReLU(A*W+B)/S_{A_2}
$$
把下一层对激活值的量化操作融合到上一层中
$$
S_M' = S_M/S_{A_2}
$$
在硬件计算时
$$
S_M' = Q_M'>>nm
$$
可以这样计算得到，nm取17
$$
Q_M' = S_M'*2^{nm}
$$
放缩系数的确定(对称量化)

以激活值为例，无符号数的量化，uint8
$$
S_A = max(A)/2^8
$$
以权重参数为例，有符号数的量化，int6
$$
S_W = max\{abs(min(W)),max(W)\}/2^{(6-1)}
$$

## TestBench

为网络中的各种数据分配存储空间

img : 图像

weight : 权重

biasm : 量化过程中所需用到的缩放因子

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608164437329.png" alt="image-20210608164437329" style="zoom:50%;" />

从文件中载入网络参数

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608164957065.png" alt="image-20210608164957065" style="zoom:50%;" />

inference 过程

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608165355230.png" alt="image-20210608165355230" style="zoom:50%;" />

与各个层的 golden_results(事先存储在二进制文件中) 对比

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608170003079.png" alt="image-20210608170003079" style="zoom:50%;" />

## SkyNet IP核

### 接口部分

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608170229631.png" alt="image-20210608170229631" style="zoom:50%;" />

相关知识参考

ug902

​	Ch1 High-Level-Synthesis

​		Managing Interfaces; Optimizing the Design 

什么是 Interface Synthesis

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608171039021.png" alt="image-20210608171039021" style="zoom:50%;" />

一般包含哪些类别的接口

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608171126285.png" alt="image-20210608171126285" style="zoom:50%;" />

其中，

Block-Level 级别的接口用来对 IP 核的运行进行控制，在 Host 部分，上层 Python 中通过不断读取 ap_ready 标志信号来判断什么时候可以向 IP 输入下一张图片，该部分接口由 Vivado HLS 自动生成

Port-Level 级别的接口用来与别的模块传输数据，如从 DDR 中加载图片和参数数据，以及将中间结果暂存到 DDR 中

另外，在 SkyNet 中由于函数的 return 被 s_axilite 形式包装，所有的 block level 接口都以 AXI4-Lite 的形式实现，方便 Zynq 上 arm 核对该 IP 的控制

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608171900696.png" alt="image-20210608171900696" style="zoom:50%;" />

其他接口类型均为 AXI4 master 类型，可将多个函数接口部署到一个 AXI4 接口上

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608172233679.png" alt="image-20210608172233679" style="zoom:50%;" />

depth

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608172449174.png" alt="image-20210608172449174" style="zoom:50%;" />

<img src="/Users/heyiqian/宽广/PYNQ/SkrNet/pics/image-20210608172549639.png" alt="image-20210608172549639" style="zoom:50%;" />

offset

- generate an offset port and automatically map it to an AXI4-Lite slave interface

allocation

<img src="pics/image-20210608172956026.png" alt="image-20210608172956026" style="zoom:50%;" />

一个简单的例子

<img src="pics/image-20210608173503621.png" alt="image-20210608173503621" style="zoom:50%;" />

### 函数主体

将四张图片拼为一张进行输入，为了能够在不同层之间共享 buffer

- 四张图片依次进行计算

- 采取 Tiling 的策略，每次计算时只计算特征图的一部分进行计算 40x80 大小的区域，对于第一个 conv 而言，输入尺寸为 160x320 故在 Width 和 Height 两个方向上各需要 4 次循环，把原图像 tiling 成 16 个子区域分次计算

  在这 16 次计算中，在每一次特定的 conv 计算前，把下一次 conv 计算所要用到的 weight 取到 buffer 中，这样**取数据的延迟和计算的延迟相互交叠**，可以缩短系统处理一张图片所需要的时间

  <img src="pics/image-20210608210118301.png" alt="image-20210608210118301" style="zoom:50%;" />

- 以 32 个通道为一组进行计算，因为对于 weight 和 activation 而言，每一个数据为 8 位，使用 256 位的总线位宽可以一次读取/写入 32 个数据；对于 bias 而言，每一个数据为 16 位，使用 256 位的总线，每次可读取/写入 16 个数据

- 流程    

  - 将本次计算的 tiling 部分加载到 FM1 中，作为本次卷积计算的输入数据
  - 将下次计算的 tiling 部分加载到 FM2 中，作为下次卷积计算的输入数据
  - 进行 3x3_conv 计算，将中间结果保存到 FM4 中
  - 进行 activation 操作和 requantization 操作，将结果保存在 FM3 中
  - 将 FM3 中的数据(进行 pooling 操作)并输出到 DDR 中暂存

- 该 IP 核中的 buffer

  - 存 FM 的 4 个
    - ADT 的 3 个
    - RDT 的 1 个
  - 存 3x3 权重的 3 个
  - 存 1x1 权重的 2 个
  - 存 bias 和 biasm 的各 3 个

<img src="pics/image-20210609143351258.png" alt="image-20210609143351258" style="zoom:50%;" />

---

#### 加载数据

- 基地址

  <img src="pics/image-20210608202500027.png" alt="image-20210608202500027" style="zoom:50%;" />

- 偏移

<img src="pics/image-20210608202606645.png" alt="image-20210608202606645" style="zoom:50%;" />

具体来看加载权重的函数

- 首先，计算要读取的权重数据的地址

- 其次，把一次从总线上读取的 256 位数据分配给对应的 32 个通道
  - 第一个 d-conv 权重 3 通道 3x3，每 32 个通道的数据组合在一起占一个地址，因此该 conv 共需要 9 个，除了前三个通道有数据，后面的通道均用 0 填充(无用数据)

<img src="pics/image-20210609100951896.png" alt="image-20210609100951896" style="zoom:50%;" />

Mx 对应输出通道的组数(不再仅仅局限于以 32 为一组，对于 p-conv 而言，一组对应 ic 个通道)；Nx对应输入通道的组数(32)；ic对应该层输入通道数

对于 p-conv weight 参数所对应地址的嵌套逻辑

- 输出通道 Mx * ic
  - 输入通道 Nx * 32
    - 每组 32 个通道中的每一个

<img src="pics/image-20210609102730781.png" alt="image-20210609102730781" style="zoom:50%;" />

加载 bias 和 biasm 数据，由于 256 位总线每次只能通过 16 个通道的数据所以需要两次总线的数据操作

对应着输出的 32 个通道

<img src="pics/image-20210609103310536.png" alt="image-20210609103310536" style="zoom:50%;" />

加载图像数据，为一个 tiling 区域(大小为 40x80)加载图片数据，对应的地址需要考虑图片数以及沿宽、高方向的偏移，只取 32 个通道中有用的 RGB 三通道信息，对于上下左右的四个边界像素以0填充

h_o 代表沿 h 方向的偏移；w_o 代表沿 w 方向的偏移

输入的 32 个通道的图片信息中，只有前 3 个通道的数据是有用的，IFM 取 \[32\]\[42\]\[82\] 就够了，输入的图片本身没有做 padding 通过这里的操作将四周 padding 了一下

padding 的左上角第一个像素对应 img 坐标为 img\[-320-1\]，每个像素位置对应一个地址，每个地址包含在该位置上 32 个通道的数据

**这里边界为什么要补 128？应该补 0 吧**

<img src="pics/image-20210609103925431.png" alt="image-20210609103925431" style="zoom:50%;" />

<img src="pics/IMG_1291.jpg" alt="IMG_1291" style="zoom:50%;" />

其中涉及到 HLS 的部分

- II(initial interval)

<img src="pics/image-20210609101726011.png" alt="image-20210609101726011" style="zoom:50%;" />

注意区分 II 和 Latency 的区别

- DATA.range()

选取特定位置区间的数据

<img src="pics/image-20210609102103016.png" alt="image-20210609102103016" style="zoom:50%;" />

<img src="pics/image-20210609102134252.png" alt="image-20210609102134252" style="zoom:50%;" />



#### 计算

##### 3x3 d-conv

输入 \[32\]\[43\]\[83\] 实际上只有 \[32\]\[42\]\[82\]中有值

输出 \[32\]\[43\]\[83\] 实际上只有 \[32\]\[1:42\]\[1:82\]中有值

- 实际上输入 FM 中后两个维度上最后一个数值在刚才的 load_image 过程中是没有被赋值的

- 通过 line_buffer 和 window_buffer 找到需要的 3x3 图像数据
- 进行 MAC9 计算，并截断溢出值
  - clip with RDT
- line_buffer 记录 3 行和所有列
- window_buffer 记录 3x3 大小区域的像素

<img src="pics/image-20210609152803182.png" alt="image-20210609152803182" style="zoom:50%;" />

HLS 中有关 array_partition 的内容

array 的分割方式

<img src="pics/image-20210609153307161.png" alt="image-20210609153307161" style="zoom:50%;" />

分割多维度 array 时，不同参数的作用

<img src="pics/image-20210609153350994.png" alt="image-20210609153350994" style="zoom:50%;" />

##### 1x1 p-conv

与 activation 一样，只对中间的 41x81 个像素有效

使用 MAC16 每次可进行 16 次乘加操作，对于 32 个通道的一组数据，需要两次计算并将两次计算结果依次进行截断相加再截断的操作

- clip with RDT

循环嵌套关系

- 输入通道数(对 d-conv 而言恒为 1)
  - 像素点(包含 width 和 height)
    - 输出通道数

<img src="pics/image-20210609154806014.png" alt="image-20210609154806014" style="zoom:50%;" />

##### activation

从上一层 p-conv 输出的数据中 \[1:42\]\[1:82\] 这些范围的 41x81 个像素有效

- 加 bias
- ReLU
- Requantization for next layer's input；clip with max&min for ADT

<img src="pics/image-20210609155810705.png" alt="image-20210609155810705" style="zoom:50%;" />

##### pooling

从上一层 p-conv 输出的数据中 \[1:41\]\[1:81\] 这些范围的 40x80 个像素有效

- 找到输出到 DDR 的地址
- 在输入中对每四个像素做 maxpooling 操作

对于输入的 FM 而言，只考虑其中间部分 40x80 的像素

对第一个 pool 操作而言 tile 为 4，即四张图片之间各有一个 padding 的存在，图片四周也有对应的 padding

<img src="pics/image-20210609160428121.png" alt="image-20210609160428121" style="zoom:50%;" />

- 没有 pooling 操作的暂存中间结果(不同层之间)

  - 边界补0
  - 计算输出地址
  - 输出数据

  对应一张图片的尺寸为 20 x 40

  其输入为 activation 层的输出，只在\[1:41\]\[1:81\] 这些范围的 40x80 个像素有值 所以需要 padding

  <img src="pics/image-20210609162940900.png" alt="image-20210609162940900" style="zoom:50%;" />

  <img src="pics/image-20210609165733967.png" alt="image-20210609165733967" style="zoom:50%;" />

  Cx 为 input_channel 的组数

  对应一张图片的尺寸为 80 x 160 或 160 x 320

  <img src="pics/image-20210609163501261.png" alt="image-20210609163501261" style="zoom:50%;" />

  参数 tile 表示每张图片由几个 tile 组成：2

  所以，每格两个 tile 是一张新的图片，需要在它们中间加一个分割区域

  <img src="pics/image-20210609165455158.png" alt="image-20210609165455158" style="zoom:50%;" />

##### reorg

32 个通道为一组，取 6 组，192 个通道的数据中 [0,0] 位置的像素作为前 192 个通道，[0,1] 为 192~192x2-1 以此类推

<img src="pics/image-20210609170058863.png" alt="image-20210609170058863" style="zoom:50%;" />

注意此处的循环顺序其实容易产生误解的

看最后一行代码，Rx 在这里起着外层循环的作用，而 Nx 起着内层循环的作用

<img src="pics/image-20210609205658504.png" alt="image-20210609205658504" style="zoom:50%;" />

##### compute_bbox

对四张图片，依次求 bbox 并通过 bias 总线进行输出

<img src="pics/image-20210609173413398.png" alt="image-20210609173413398" style="zoom:50%;" />

在 main 函数中做后续操作得到 bbox 的归一化值

<img src="pics/image-20210609173959628.png" alt="image-20210609173959628" style="zoom:50%;" />

