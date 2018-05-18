# 圖像訓練：使用卷積神經網路辨識手寫數字

在這份教學中，我們將建立一個 TensorFlow.js 模型，並使用卷積神經網路（CNN）來分類手寫數字。首先，我們先透過讓模型「看」幾千個手寫數字和其標籤來訓練我們的分類器。然後我們將使用這個模型從沒看過的測試資料來測試這個分類器的準確率。

## 開始之前

這份教學假設你已經熟悉 TensorFlow.js 的基本架構（張量、變數和運算子），以及優化、損失的概念。有關於這些主題的更多資訊，我們建議你先閱讀這些之前的教學再閱讀這份教學：

- [TensorFlow.js 的核心概念](/tutorials/core-concepts.md)
- [訓練的第一步：從合成數劇中訓練曲線](/tutorials/fit-curve.md)

## 執行程式碼

這份教學的完整代碼可以在 [TensorFlow.js 範例](https://github.com/tensorflow/tfjs-examples/tree/master/mnist) 中的 [tfjs-examples/mnist](https://github.com/tensorflow/tfjs-examples/tree/master/mnist) 資料夾找到。

你可以透過複製這個專案並執行這個範例：

```bash
$ git clone https://github.com/tensorflow/tfjs-examples
$ cd tfjs-examples/mnist
$ yarn
$ yarn watch
```

在上面的專案中，[tfjs-examples/mnist](https://github.com/tensorflow/tfjs-examples/tree/master/mnist) 資料夾是完全獨立的，所以你可以複製它並開始自己的專案。

**註：**這份教學和 [tfjs-examples/mnist-core](https://github.com/tensorflow/tfjs-examples/tree/master/mnist) 範例中的差別是我們在這裡使用了 TensorFlow.js 的高階 API（`Model`、`Layer`） 來建構模型；而 [mnist-core](https://github.com/tensorflow/tfjs-examples/tree/master/mnist) 使用低階的線性代數運算子來建立類神經網路。

## 資料

我們在本教學中將使用 MNIST 手寫資料集，這些即將拿來訓練分類器的手寫資料看起來像這樣：

![MNIST 4](/images/mnist_4.png) ![MNIST 3](/images/mnist_3.png) ![MNIST 8](/images/mnist_8.png)

我們使用 [data.js](https://github.com/tensorflow/tfjs-examples/blob/master/mnist-core/data.js) 來預處理我們的資料。其中包含了一個類別 `MnistData`，方便我們從 MNIST 託管資料集中取得隨機批次的 MNIST 圖片。

`MnistData` 將整個資料及分成訓練資料和測試資料。當我們訓練模型時，分類器只會看到訓練集；當我們評估模型時，我們只會使用模型未曾借過的測試資料，以查看我們的模型如何預測這些全新資料。

`MnistData` 有兩個公開方法：

- `nextTrainBatch(batchSize)`：從訓練集中回傳一批隨機圖像及其標籤
- `nextTestBatch(batchSize)`：從測試集中回傳一批圖像及其標籤

**註：**在訓練 MNIST 分類器時，隨機拿出資料非常重要，這樣模型訓練才不會因為我們提供的順序而受影響。例如我們先將所有 *1* 丟進去，在此訓練階段中可能學會很簡單的預測 1（因為這會最小化損失）。如果我們只餵 2 給模型，它可能會容易只預測出 2 而永遠不會是 1（因為這又會最小化損失）。我們的模型將無法學習對代表性的數字樣本作出準確的預測。

## 建立模型

在這一節，我們將建立一個卷積圖像分類模型。我們會用一個 `Sequential` 模型（最簡單的一種模型），其中張量將一層一層傳遞到下一層。

首先，讓我們先用　`tf.sequential` 產生一個 `Sequential` 模型實體。

```javascript
const model = tf.sequential();
```

現在我們已經建立了一個模型，讓我們在模型裡加上層。

## 增加第一個層

我們要增加的第一個層是一個二維卷積層。卷積在圖片上滑動率波來學習不同空間的變形（即圖片上的不同模型或物件將以相同的方式處理）。有關卷積的更多訊息，請看 [這篇文章](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)。

我們可以使用 [`tf.layers.conv2d`](https://js.tensorflow.org/api/latest/index.html#layers.conv2d) 來建立二維卷積層，而這個卷積層可以接受定義層架構的參數：

```javascript
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling'
}));
```

讓我們來看看物件裡的每個參數：

- `inputShape`：流入模型第一層的資料的形狀。在這個範例中，我們的 MNIST 範例是 28x28 像素的黑白圖片。圖片資料的規範格式是 `[row, column, depth]`，所以我們在這裡想要為每個維度的像素數配置 `[28, 28, 1]`── 28 個行和列作為每一個維度的像素，而深度用 1 是因為我們只有一種顏色。
- `kernelSize`：要用於資料的滑動卷積濾波器的窗口大小。在這裡我們設置一個 `5` 的 `kernelSize`，指定一個 5x5 的卷積窗口。
- `filters`：大小為 `kernelSize` 的篩選器窗口的數量，以應用於輸入資料。在這裡我們對資料用 8 個過濾器。
- `strides`：滑動窗口的「步數」。即每次在圖案上移動時該移動多少像素。我們在這裡使用 1 代表過濾器每次將滑動 1 個像素為單位。
- `activation`：卷積完成後應用於資料的 [激勵函數](https://developers.google.com/machine-learning/glossary/#activation_function)。這裡我們使用 [整流線性單元（ReLU）](https://developers.google.com/machine-learning/glossary/#ReLU) 方法，這是機器學習模組中常見的激勵函數。
- `kernelInitializer`：用於初始化模型權重的方法，它對於訓練動態非常重要。我們在這裡不會詳細介紹初始化，但這裡使用的 `VarianceScaling` 是一般來說很棒的初始器選擇。

## 增加第二層

讓我們在模型中增加第二個層：一個最大池化層。我們會使用 [`tf.layers.maxPooling2d`](https://js.tensorflow.org/api/latest/index.html#layers.maxPooling2d)。該層會透過計算每個滑動窗口的最大值來縮減卷積結果（又稱為激勵）：

```javascript
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
```

我們來看一下這些參數：

- `poolSize`：設定輸入資料的滑動窗口大小。在這邊我們設 `poolSize` 為 `[2, 2]`，表示池化層將用 2x2 的窗口套用在輸入資料上。
- `strides`：滑動窗口的「步數」。即每次在圖案上移動時該移動多少像素。在這裡我們使用 `[2, 2]`，表示每次篩選器都會在圖片上以 2 像素為單位垂直和水平移動。

**註：**由於 `poolSize` 和 `strides` 都是 2x2，池窗口將會完全不重疊。這代表池化層會將前一層的激活大小減半。

## 增加剩餘層

重複層結構是神經網路中常見的模式。讓我們在模型裡再增加第二個卷積層，然後再增加一個池化層。注意我們的第二個卷積層裡，我們把過濾器的數量從 8 調到 16。另外還要注意我們沒有特別指定 `inputShape`，因為它可以從上一層的輸出自己推裡出來：

```javascript
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling'
}));

model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
```

接著，我們增加一個 [`flatten`](https://js.tensorflow.org/api/latest/index.html#layers.flatten) 層來把前一層的輸出平整化成向量：

```javascript
model.add(tf.layers.flatten());
```

最後，讓我們增加一個 [`dense`](https://js.tensorflow.org/api/latest/index.html#layers.dense) 層（又稱為全連接層），它會執行最後的分類。在全連接層之前把卷積層和池化層平整化輸出也是另外一個類神經網路中常見的模式：

```javascript
model.add(tf.layers.dense({
  units: 10,
  kernelInitializer: 'VarianceScaling',
  activation: 'softmax'
}));
```

讓我們詳細看一下丟進 `dense` 層裡的參數：
- `units`：輸出激勵的大小。由於這是最後一層，而我們正在做一個 10 級別的分類任務（數字 0~9），我們在這裡用 10 單位。（有時候單位指的是*神經元*的數量，但我們避免使用這個術語。）
- `kernelInitializer`：我們使用和卷積層一樣的 `VarianceScaling` 初始化方法
- `activation`：分類任務的最後一層激勵方法通常會使用 [softmax](https://developers.google.com/machine-learning/glossary/#softmax)。它將我們的 10 維輸出正規化成機率分不，所以我們會有 10 個類別中每個類別的機率。

## 訓練模型

為了真正訓練模組，我們需要建構一個優化器並定義損失函數。我們還要定義評估指標來衡量我們的模型在資料上的表現。

**註：**想要深入了解 TenslorFlow.js 中的優化器和損失函數，請閱讀 [Training First Steps](/tutorials/fit-curve.md)。

### 定義優化器

在我們的卷積神經網路模型中，我們使用 [隨機梯度下降法（SGD）](https://developers.google.com/machine-learning/glossary/#SGD) 當作我們的優化器，學習率為 0.15。

```
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
```

### 定義損失

在我們的損失函數中，我們使用交叉熵（cross-entropy）（`categoricalCrossentropy`）。它通常用來優化分類任務。`categoricalCrossentropy` 衡量了我們模型裡最後一層產生出來的機率分佈和我們標籤中給定的機率分佈間的誤差，這個分佈在正確的標籤會是 1（%）。例如，以下是數字 7 給定的標籤和預測值的範例：

|類別 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----------|---|---|---|---|---|---|---|---|---|---|
|標籤 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
|預測|.1 |.01|.01|.01|.20|.01|.01|.60|.03|.02|

如果有高機率是 7，`categoricalCrossentropy` 會給出比較低的損失值；而如果有低機率是 7，它就會給出比較高的損失值。在訓練中，模組會更新內部的參數以最小化整個資料集裡的 `categoricalCrossentropy`。

### 定義評估指標

在我們的評估指標，我們會使用準確性（accuracy）：即正確預測與所有預測的比值。

### 編譯模型

為了編譯模組，我們傳一個參數物件給我們的模組，物件裡包含優化器、損失函數和一份評估指標（在這裡我們只使用 `accuracy`）：

```javascript
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
```

### 設置批次大小

在開始訓練之前，我們需要再定義一些有關批次大小的參數：

```javascript
// 模組進行參數更新時應該要「看」幾個範例
const BATCH_SIZE = 64;
// 要用幾個批次訓練模組
const TRAIN_BATCHES = 100;

// 每 TEST_ITERATION_FREQUENCY 個批次，就檢查 TEST_BATCH_SIZE 的準確性
// 理想上我們應該算整個測試資料的準確性，但為了效能，我們將只測試子資料集。
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
```

**關於有關分批和批次大小**

為了充分利用 GPU 並行計算的能力，我們想要將多個批次一起輸入網路並使用簡單的前饋修正。

使用批次計算的另一個原因是在優化過程中，我們只在多個範例進行梯度平均後才進行內部參數修正（花一個步驟），這讓我們避免因為一個錯誤的例子（例如標籤錯誤）就走到錯的方向。

當批次輸入資料時，我們引入 rank D+1 的張量，其中 D 是一個單一輸入的維度數量。

如前面所討論的，我們 MNIST 資料集中的每個單一圖片維度為 `[28, 28, 1]`。當我們將 `BATCH_SIZE` 設為 64 時，表示我們一次批次處理 64 個圖片，也就是說我們資料的實際形狀是 `64, 28, 28, 1`。（批次永遠是最外面的維度）

**註：**回想我們第一次在 `conv2d` 設定的 `inputShape` 沒有指定批次大小（64）。這些參數裡的批次大小被寫成不可知的，所以他們可以接受任意大小的批次。

### 撰寫訓練循環

以下是訓練迴圈的程式碼：

```javascript
for (let i = 0; i < TRAIN_BATCHES; i++) {
  const batch = data.nextTrainBatch(BATCH_SIZE);

  let testBatch;
  let validationData;
  // 每幾個批次就測試準確性
  if (i % TEST_ITERATION_FREQUENCY === 0) {
    testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
    validationData = [
      testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
    ];
  }

  // 整個資料集並沒有被載入記憶體裡，所以我們重複呼叫這些批次
  const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
      batch.labels,
      {
        batchSize: BATCH_SIZE,
        validationData,
        epochs: 1
      });

  const loss = history.history.loss[0];
  const accuracy = history.history.acc[0];

  // ... 繪製圖表 ...
}
```

讓我們仔細看看這些程式碼。首先我們取得了一批訓練範例。回想我們之前利用 GPU 並行計算的優點把範例批次傳入，而且在訓練許多個範例後才平均更新這些參數：

```javascript
const batch = data.nextTrainBatch(BATCH_SIZE);
```

每 5 步（`TEST_ITERATION_FREQUENCY`），我們就建立一個 `validationData`——一個有兩個元素，包含測試資料集裡的一批 MNIST 資料的陣列和他們的對應標籤。我們將會使用這些資料來評估這個模組的準確性：

```javascript
if (i % TEST_ITERATION_FREQUENCY === 0) {
  testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
  validationData = [
    testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]),
    testBatch.labels
  ];
}
```

`model.fit` 即是我們訓練模組且實際更新參數的地方

**註：＊＊在整個資料集上呼叫 `model.fit()` 會導致整個資料集上傳到 GPU，甚至凍結整個應用程式。為了避免一次丟太多資料到 GPU，我們建議在 `for` 迴圈裡呼叫 `model.fit()`，每次只傳遞一批資料。如下所示：

```javascript
// 整個資料集並沒有被載入記憶體裡，所以我們重複呼叫這些批次
  const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
      {batchSize: BATCH_SIZE, validationData: validationData, epochs: 1});
```

接著再讓我們仔細看看這些參數：

- `x`：我們的圖片資料。記得我們是批次傳遞範例，所以我們必須告訴 `fit` 方法這批次有多大。`MnistData.nextTrainBatch` 回傳圖片及形狀 `[BATCH_SIZE, 784]`——所有影像資料都是長度為 784（28*28）的一維向量。然而我們的模組預期的是形狀為 `[BATCH_SIZE, 28, 28, 1]` 的形狀，所以我們必須重新變形（[`reshape`](https://js.tensorflow.org/api/latest/index.html#reshape)）。
- `y`：我們的標籤，也就是每個圖片的實際數字分類。
- `batchSize`：每次丟進去訓練的批次有幾張圖。稍早我們把 `BATCH_SIZE` 設成 64 了。
- `validationData`：表示每 `TEST_ITERATION_FREQUENCY` 次迭代就進行一次驗證（此處是 5），這些資料的形狀為 `[TEST_BATCH_SIZE, 28, 28, 1]`。稍早我們已將 `TEST_BATCH_SIZE` 設為 1000，我們的評估指標（準確性）就會用這樣子的資料集大小。
- `epochs`： 每批次要執行訓練的次數。由於我們迭代把批次送進去訓練，我們只需要從這個批次訓練一次即可。

每當我們呼叫 `fit` 的時候，它會回傳一個豐富的物件，包含存放在 `history` 的指標的記錄（logs）。我們在每次訓練迭代都提取一次我們的損失和準確性，所以我們可以把他們畫在圖表上：

```javascript
const loss = history.history.loss[0];
const accuracy = history.history.acc[0];
```

## 查看結果！

如果你執行完整的程式碼，你會看到類似這樣的輸出：

![兩張圖表：第一張顯示損失下降、第二張準確性提升。](/images/mnist_learned.png)

看起來這模型可以成功預測大部分的圖片的數字！做得好！

## 其他資源

- 更多卷積網路，請查看 Chris Olah 的 [Unerstanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)。
- 更多關於損失，請看 [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) 的 [Descending into ML](https://developers.google.com/machine-learning/crash-course/descending-into-ml/) 來深入了解機器學習損失。
- 更多梯度下降和 SGD，請查看 [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) 的 [Reducing Loss](https://developers.google.com/machine-learning/crash-course/reducing-loss/)。