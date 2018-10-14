# TensorFlow.js 的核心概念

**TensorFlow.js** 是一個開源、WebGL 加速的 JavaScript 機器學習套件。它將高性能的機器學習模型帶到你的指尖、讓你在瀏覽器上訓練類神經網路或在推理模式下使用預先訓練好的模型。閱讀 [Getting Started](#) 來了解如何安裝／設定 TensorFlow.js

TensorFlow.js 提供機器學習的低階模型以及以 Keras 為靈感的高階 API 來建立神經網路。讓我們一起看套件裡的核心元件。

## 張量

在 TensorFlow.js 中，資料的核心單位是張量（Tensor）：一組形成一維或多維陣列的數值。一個 [Tensor](https://js.tensorflow.org/api/latest/index.html#class:Tensor) 實體有 `shape` 屬性，定義了陣列的形狀（即陣列的每個維度裡有幾個值）。

主要的 `Tensor` 建構子是 [`tf.sensor`](https://js.tensorflow.org/api/latest/index.html#tensor)：

```javascript
// 2x3 的張量
const shape = [2, 3]; // 2 行 3 列
const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
a.print(); // 印出張量的值
// 輸出： [[1 , 2 , 3 ],
//        [10, 20, 30]]

// 形狀也可以被推理出來：
const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
b.print();
// 輸出： [[1 , 2 , 3 ],
//        [10, 20, 30]]
```

然而，建構低階張量時，我們推薦使用以下方法來增加程式碼的閱讀性：[`tf.scalar`](https://js.tensorflow.org/api/latest/index.html#scalar)、[`tf.tensor1d`](https://js.tensorflow.org/api/latest/index.html#tensor1d)、[`tf.tensor2d`](https://js.tensorflow.org/api/latest/index.html#tensor2d)、[`tf.tensor3d`](https://js.tensorflow.org/api/latest/index.html#tensor3d) 和 [`tf.tensor4d`](https://js.tensorflow.org/api/latest/index.html#tensor4d)。

下面是一個範例，用 `tf.tensor2d` 建立了一個和上面一樣的張量：

```javascript
const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
c.print();
// 輸出： [[1 , 2 , 3 ],
//        [10, 20, 30]]
```

TensorFlow.js 同時也提供了方便的方法來建立全部值為 0 的張量（[`tf.zeros`](https://js.tensorflow.org/api/latest/index.html#zeros)）或全部為 1 的張量（[`tf.ones`](https://js.tensorflow.org/api/latest/index.html#ones)）：

```javascript
// 3x5 張量，值都為 0
const zeros = tf.zeros([3, 5]);
// 輸出： [[0, 0, 0, 0, 0],
//        [0, 0, 0, 0, 0],
//        [0, 0, 0, 0, 0]]
```

在 TensorFlow.js 中，張量是不可改變的（immutable），一旦建立，你就不能再改變它的值；反之，你可以對他們執行操作來產生新的張量。

## 變數

變數（[`Variable`s](https://js.tensorflow.org/api/latest/index.html#class:Variable)） 用一個張量的值來初始化。不像張量，變數的值是可以改變的。你可以使用 `assign` 方法來指派一個新張量的值放在已經存在的變數：

```javascript
const initialValues = tf.zeros([5]);
const biases = tf.variable(initialValues); // 初始化偏差
biases.print(); // 輸出： [0, 0, 0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
biases.assign(updatedValues); // 更改偏差的值
biases.print(); // 輸出： [0, 1, 0, 1, 0]
```

變數主要用來儲存並在訓練模型是更新數值。

## 運算子

張量可以讓你儲存資料，而運算子可以讓你操控資料。TensorFlow.js 提供了各式各樣適合線性代數和機器學習的運算子，能套用在張量上。由於張量不可改變，這些運算子不會改變他們的值，而會返回新的張量。

可用的運算子包含一元運算子如 [`square`](https://js.tensorflow.org/api/latest/index.html#square)：

```javascript
const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// 輸出： [[1, 4 ],
//        [9, 16]]
```

而二元運算子像 [`add`](https://js.tensorflow.org/api/latest/index.html#add)、[`sub`](https://js.tensorflow.org/api/latest/index.html#sub)、[`mul`](https://js.tensorflow.org/api/latest/index.html#mul)：

```javascript
const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// 輸出： [[6 , 8 ],
//        [10, 12]]
```

TensorFlow.js 有鏈結 API；你可以在運算子的結果再次呼叫運算子：

```javascript
const sq_sum = e.add(f).square();
sq_sum.print();
// 輸出： [[36 , 64 ],
//        [100, 144]]

// 所有的運算方法也為主要命名空間中的函數公開
// 所以你可以這樣操作：
const sq_sum = tf.square(tf.add(e, f));
```

## 模型與層

概念上，模型（Model）是一個給定輸入就會產生預期輸出的方法。

在 TensorFlow.js 中有*兩個*建立模型的方法，你可以*直接使用運算子*來代表模型運作的方法，例如：

```javascript
// 定義函數
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // 下個部分會再詳細講 tf.tify
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// 定義係數： y = 2x^2 + 4x + 8
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

// 輸入 2 ，預測輸出
const result = predict(2);
result.print() // 輸出：24
```

你也可以用深度學習中比較熱門的方法：透過高階 API [`tf.model`](https://js.tensorflow.org/api/latest/index.html#model) 用*層*建構模型。下面這段程式碼建構了一個 [`tf.sequential`](https://js.tensorflow.org/api/latest/index.html#sequential) 的模型：

```javascript
const model = tf.sequential();
model.add(
  tf.layers.simpleRNN({
    units: 20,
    recurrentInitializer: 'GlorotNormal',
    inputShape: [80, 4]
  })
);

const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({optimizer, loss: 'categoricalCrossentropy'});
model.fit({x: data, y: labels)});
```

TensorFlow.js 中還有許多可用的不同型態的層，像是 [`tf.layers.simpleRNN`](https://js.tensorflow.org/api/latest/index.html#layers.simpleRNN)、[`tf.layers.gru`](https://js.tensorflow.org/api/latest/index.html#layers.gru)、[`tf.layers.lstm`](https://js.tensorflow.org/api/latest/index.html#layers.lstm)。

## 記憶體管理：dispose 和 tf.tidy

由於 TensorFlow.js 使用 GPU 來加速數學運算，使用張量和變數時管理 GPU 記憶體是必要的。

TensorFlow.js 提供了兩個方法來幫助這個：`dispose` 和 [`tf.tidy`](https://js.tensorflow.org/api/latest/index.html#tidy)。

### Dispose

你可以呼叫在張量或變數上 `dispose` 以清除它，並釋放 GPU 記憶體：

```javascript
const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
const x_squared = x.square();

x.dispose();
x_squared.dispose();
```

### tf.tidy

進行大量張量操作時，使用 `dispose` 可能變得有點麻煩。TensorFlow.js 提供了另外一個方法：`tf.tidy`。它可以在 GPU 端的張量做到類似 JavaScript 中區域（regular scope）的作用。

```javascript
// tf.tidy 執行一個方法，並在最後清理它。
const average = tf.tidy(() => {
  // tf.tidy 會清理這方法內所有被張量用掉的記憶體，除了需要的張量以外。
  //
  // 即使是像下面的簡單操作，一些中間產物張量也會產生。所以保持簡潔的數學運算是很重要的。
  const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
  const z = tf.ones([4]);

  return y.sub(z).square().mean();
});

average.print() // 輸出： 3.5
```

使用 `tf.tidy` 能避免你的應用程式記憶體洩漏，也可以用來更仔細的控制何時回收記憶體。

#### 兩件重要的事情

- 丟進 `tf.tidy` 的方法應該要是同步的且不能返回一個 Promise。我們建議不要把更新 UI 或遠端請求的程式碼放進 `tf.tidy`。
- `tf.tidy` 不會清理變數。變數通常持續到整個生命週期或機器學習模型，所以就算變數被放進 `tf.tidy`，TensorFlow.js 也不會清理它們；然而你可以手動呼叫 `dispose`。

# 其他資源

造訪 [TensorFlow.js API Reference](https://js.tensorflow.org/api/latest/index.html) 以查看套件的詳細文件。

想要更深度的瞭解機器學習架構，可以閱讀以下資源：
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)（註： 這個課程的練習使用 TensorFlow 的 [Python API](https://www.tensorflow.org/api_docs/python/)，不過機器學習的核心概念一樣可以應用在 TensorFlow.js）
- [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
