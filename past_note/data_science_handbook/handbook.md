[toc]

---

# handbook

---

## ipython
### shortcuts
### magic

```python
1. %run
		%run pyfilm.py
2. %timeit(运行多次)
		%timeit 单行
		%%timeit 代码块
		
		%%timeit
		L = []
		for n in range(1000):
			L.append(n ** 2)
		
		%time L.sort()
		%%time # 代码块
3. %lsmagic and %magic
4. %history
		%history -n 1-4
```
### In/Out

### shell
ipython中一行中任何在 ! 之后的内容,将不会通过 Python 内核运行，而是通过系统命令行运行
shell是一种通过文本与计算机交互的方式。

### 调试 异常
%xmode 可以在轨迹追溯（traceback）中找到引起这个错误的原因
- %xmode 有一个输入参数，即模式。模式有 3 个可选项：Plain、Context 和 Verbose。默认情况下是 Context。Plain 更紧凑，给出的信息更少,Verbose 模式加入了一些额外的信息，包括任何被调用的函数的参数

%debug

- 它会在异常点自动打开一个交互式调试提示符,输入 quit 来结束调试会话

### 分析
1. %prun func(x)
2. %lprun 
	逐行分析
3. 用%memit和%mprun进行内存分析

## numpy
NumPy 要求数组必须包含同一类型的数据。如果类型不匹配，NumPy 将会向上转换（如果可行）

### 基本操作
```python
np.array([1, 4, 2, 5, 3])
np.array([1, 2, 3, 4], dtype='float32')
np.array([range(i, i + 3) for i in [2, 4, 6]])

# 创建一个长度为10的数组，数组的值都是0
np.zeros(10, dtype=int)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 创建一个3×5的浮点型数组，数组的值都是1
np.ones((3, 5), dtype=float)
array([[ 1., 1., 1., 1., 1.],
[ 1., 1., 1., 1., 1.],
[ 1., 1., 1., 1., 1.]])

# 创建一个3×5的浮点型数组，数组的值都是3.14
np.full((3, 5), 3.14)
array([[ 3.14, 3.14, 3.14, 3.14, 3.14],
[ 3.14, 3.14, 3.14, 3.14, 3.14],
[ 3.14, 3.14, 3.14, 3.14, 3.14]])

# 创建一个3×5的浮点型数组，数组的值是一个线性序列
# 从0开始，到20结束，步长为2（它和内置的range()函数类似）
np.arange(0, 20, 2)
array([ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

# 创建一个5个元素的数组，这5个数均匀地分配到0~1
np.linspace(0, 1, 5)
array([ 0. , 0.25, 0.5 , 0.75, 1. ])

# 创建一个3×3的、在0~1均匀分布的随机数组成的数组
np.random.random((3, 3))
array([[ 0.99844933, 0.52183819, 0.22421193],
[ 0.08007488, 0.45429293, 0.20941444],
[ 0.14360941, 0.96910973, 0.946117 ]])

# 创建一个3×3的、均值为0、方差为1的
# 正态分布的随机数数组
np.random.normal(0, 1, (3, 3))
array([[ 1.51772646, 0.39614948, -0.10634696],
[ 0.25671348, 0.00732722, 0.37783601],
[ 0.68446945, 0.15926039, -0.70744073]])

# 创建一个3×3的、[0, 10)区间的随机整型数组
np.random.randint(0, 10, (3, 3))
array([[2, 3, 4],
[5, 7, 8],
[0, 5, 0]])

# 创建一个3×3的单位矩阵
np.eye(3)
array([[ 1., 0., 0.],
[ 0., 1., 0.],
[ 0., 0., 1.]])

# 创建一个由3个整型数组成的未初始化的数组
# 数组的值是内存空间中的任意值
np.empty(3)
array([ 1., 1., 1.])
```

### 数据类型
- 数据类型描述
bool_ 	布尔值（真、True 或假、False），用一个字节存储
int_ 	默认整型（类似于 C 语言中的 long，通常情况下是 int64 或 int32）
intc 	同 C 语言的 int 相同（通常是 int32 或 int64）
intp 	用作索引的整型（和 C 语言的 ssize_t 相同，通常情况下是 int32 或int64）
int8 	字节（byte，范围从–128 到 127）
int16 	整型（范围从–32768 到 32767）
int32 	整型（范围从–2147483648 到 2147483647）
int64 	整型（范围从–9223372036854775808 到 9223372036854775807）
uint8 	无符号整型（范围从 0 到 255）
uint16 	无符号整型（范围从 0 到 65535）
uint32 	无符号整型（范围从 0 到 4294967295）
uint64 	无符号整型（范围从 0 到 18446744073709551615）
float_ 	float64 的简化形式
float16	半精度浮点型：符号比特位，5 比特位指数（exponent），10 比特位尾数（mantissa）
float32	单精度浮点型：符号比特位，8 比特位指数，23 比特位尾数
float64 双精度浮点型：符号比特位，11 比特位指数，52 比特位尾数
complex_     complex128 的简化形式
complex64   复数，由两个 32 位浮点数表示
complex128  复数，由两个 64 位浮点数表示