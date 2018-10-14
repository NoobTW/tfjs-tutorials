# 訓練的第一步：從合成數據中訓練曲線

在這份教學中，我們將使用 TensorFlow.js 來將曲線調整成合成數據。給定一些添加噪音的多項式函數生成的資料，我們將訓練模型並產生多項式函數的係數。

## 開始之前

這份教學假設你已經熟悉 TensorFlow.js [核心概念](/tutorials/core-concepts.md) 裡的基本架構：張量、變數和運算子。我們推薦在閱讀這份教學之前先閱讀核心概念。

## 執行程式碼

這份教學重點在使用 TensorFlow.js 建立模型並學習其係數的程式碼。本教學的完整程式碼（包括產生數據和圖表的程式碼）可以在 [這裡](https://github.com/tensorflow/tfjs-examples/tree/master/polynomial-regression-core) 找到。

在本機執行程式碼，你需要安裝以下依賴項：

- [Node.js](https://nodejs.org) 8.9或更高版本
- [Yarn](https://yarnpkg.com/en/) 或 [NPM CLI](https://docs.npmjs.com/cli/npm)

以下的指令使用 Yarn，但如果你熟悉並選擇使用 NPM CLI 亦可：

```bash
$ git clone https://github.com/tensorflow/tfjs-examples
$ cd tfjs-examples/polynomial-regression-core
$ yarn
$ yarn watch
```

在上面的專案中，[tfjs-examples/polynomial-regression-core](https://github.com/tensorflow/tfjs-examples/tree/master/polynomial-regression-core) 資料夾是完全獨立的，所以你可以複製它並開始自己的專案。

## 輸入資料

我們的合成資料由 x 座標和 y 座標組成，繪製在笛卡爾平面上會看起來像這樣子：

![輸入資料散佈圖，資料接近三次函數，極值點最小值大約在 (-0.6, 0)、極值點最大值大約在 (0.4, 1)](/images/fit_curve_data.png)

這資料是使用三次函數：![三次函數：y=ax3+bx2+cx+d](/images/polynomial1.gif) 產生的。

而我們的任務是學習這函數的*係數*：*a*、*b*、*c*、*d* 的最適合值。讓我們來看看如何使用 TensorFlow.js 操作來學習這些值。

## 第一步：設置變數

首先讓我們設置一些變數來保持每一個步驟中我們對這些值的最佳估計。一開始我們把它們都指派一個隨機值：

```
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
```

## 第二步：建立模型

我們可以透過鏈結一系列數學運算子：加（[add](https://js.tensorflow.org/api/latest/index.html#add)）、乘（[mul](https://js.tensorflow.org/api/latest/index.html#mul)）及指數（次方（[pow](https://js.tensorflow.org/api/latest/index.html#pow)）和平方（[square](https://js.tensorflow.org/api/latest/index.html#square)））來表示我們的多項式函數 ![三次函數：y=ax3+bx2+cx+d](/images/polynomial1.gif)。

下面的程式碼建立了一個 `predict` 函數，並以 `x` 為輸入、`y` 為輸出：

```javascript
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}
```

讓我們繼續使用我們在步驟 1 設置的 *a*、*b*、*c* 和 *d*，我們的圖可能畫出來像這樣：

![三次函數不適合圖上的數據，在 x=-1.0 到 x=0 的時候太高；在 x=0.2 到 x=1.0 時圖形向上、原始數據向下。](/images/fit_curve_random.png)

## 第三步：訓練模型

我們的最後一步是訓練模型並學習良好的係數。為了訓練我們的模型，我們需要定義三個東西：

- 損失函數（Loss function）：用於衡量給定多項式和數據的吻合程度。損失值越低，多項式越吻合數據。
- 優化器（Optimizer）：根據損失函數的輸出執行係數修正的算法。優化器的目標是讓損失函數的輸出值最小。
- 訓練循環（Training Loop）：迭代執行優化器，使損失最小。

### 定義損失函數

在這份教學裡，我們將使用 [均方誤差（MSE）](https://developers.google.com/machine-learning/crash-course/glossary/#MSE) 來當我們的損失函數。均方誤差把每個實際的 y 值和我們透過每個 x 算出來的 y 值相減做平方，並取平均當作結果。

我們可以在 TensorFlow.js 中像這樣定義出一個 MSE 函數：

```javascript
function loss(predictions, labels) {
  // 用預測結果減掉實際值（label），平方並取得平均。
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}
```

### 定義優化器

對於優化器，我們將使用 [隨機梯度下降法（SGD）](https://developers.google.com/machine-learning/crash-course/glossary#SGD)。隨機梯度下降法透過取得資料集中的隨機點的 [梯度](https://developers.google.com/machine-learning/crash-course/glossary#gradient)，並使用其值來增加或減少我們模型中的係數。

TensorFlow.js 提供了一個方便方法來執行 SGD，所以您不需要擔心怎麼自己定義 SGD 中的所有數學操作。輸入一個期望的學習率，[`tf.train.sgd`](https://js.tensorflow.org/api/latest/index.html#train.sgd) 就會返回一個 `SGDOptimizer` 物件，這個物件可以被用來優化損失函數的值。

*學習率*控制了模型中改善預測的調整量。低學習率會導致整個學習過程跑得比較慢（訓練中必須使用更多的迭代次數來學習好的係數）；而高學習率會加快學習但會讓模型的結果在正確值附近擺盪、過度矯正。

以下的程式碼建立了一個學習率為 0.5 的 SGD 優化器：

```javascript
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
```

### 定義訓練循環

我們已經定義了我們的損失函數和優化器，現在我們要建立一個訓練循環。訓練循環迭代地執行 SGD 並改進我們模型中的係數以降低損失（MSE）。以下是我們訓練循環的樣子：

```javascript
function train(xs, ys, numIterations = 75) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}
```

讓我們仔細、一步一步看過這些程式碼。首先我們定義了一個訓練方法，使用我們資料集中的 x、y 值，以及預期的迭代次數當作輸入：

```javascript
function train(xs, ys, numIterations) {
...
}
```

接著如同剛剛的說明，我們定義學習率並產生一個新的 SGD 優化器。

```javascript
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
```

最後我們設置一個 `for` 迴圈，來迭代訓練 `numIterations` 次。每次迭代中，我們都會呼叫優化器中的 [`minimize`](https://js.tensorflow.org/api/latest/index.html#class:train.Optimizer)，這就是施展魔法的地方：

```javascript
for (let iter = 0; iter < numIterations; iter++) {
  optimizer.minimize(() => {
    const predsYs = predict(xs);
    return loss(predsYs, ys);
  });
}
```

`minimize` 會做兩件事情：

1. 它使用第二步定義的 *predict* 方法預測所有給定 *x* 值的 *y* 值（`predYs`）。
2. 它使用我們剛剛在**定義損失函數**中定義的損失函數算出這些預測的均方誤差。

`minimize` 會自動調整這個方法中使用的任何變數（`Variable`）（本例就是這些係數：*a*、*b*、*c* 和 *d*）來最小化回傳值（損失）。

訓練迴圈完成後，我們的 *a*、*b*、*c* 和 *d* 將會是 SGD 模型迭代 75 次的學習完的係數值。

## 查看結果！

一旦我們的程式跑完，我們就能使用 *a*、*b*、*c*、*d* 的結果來繪製曲線：

![一個三次曲線，幾乎逼進我們原始資料的形狀。](/images/fit_curve_learned.png)

這結果比我們當初隨機產生係數所繪製的曲線好很多。

## 其他資源

- 查看 [TensorFlow.js 中的核心概念](/tutorials/core-concepts.md) 來了解基本架構：張量、變數和運算子。
- 查看 [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) 中的 [Descending into ML](https://developers.google.com/machine-learning/crash-course/descending-into-ml/) 深入了解機器學習損失。
- 查看 [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) 中的 [Reducing Loss](https://developers.google.com/machine-learning/crash-course/reducing-loss/) 來了解梯度下降和 SGD。
