# 給 Keras 使用者的 TensorFlow.js 層 API
TensorFlow.js 的層 API 以 Keras 的形式形成，您可能已經從[教學](https://js.tensorflow.org/tutorials/index.html)與範例中意識到，我們考慮到 JavaScript 與 Python 之間的差異，正努力使層 API 與 Keras 類似。這使得擁有開發 Python 中 Keras 模型經驗的用戶可以輕易地轉移到 JavaScript 中的 TensorFlow.js  層。以下為 Keras 的程式翻譯成 JavaScript：

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))

```
```javascript
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();


```
然而，我們想在此文件中提出並解釋當中的差異性，一旦你理解這些差異以及其背後的基本原理，你的 Python 到 JavaScript 轉移（或反向轉移）將會是一個相對順利的體驗。

## 建構函數將 JavaScript 物件作為架構
比較以上範例中 Python 與 JavaScript 兩行：它們都建立[密集](https://keras.io/layers/core/#dense)層。

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```javascript
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```
JavaScript 函數沒有同等於 Python 函數中的關鍵字參數。我們希望避免建構函數選項以 JavaScript 中的位置參數執行，這對於具有大量關鍵字參數（[LSTM](https://keras.io/layers/recurrent/#lstm)）的建構函數來記憶和使用會特別麻煩。這就是我們使用 JavaScript 物件結構的原因。這些物件提供了與 Python 關鍵字參數相同的位置不變性與靈活性。 

有一些 Model 類別的方法，例如：`Model.compile()`也將 JavaScript 物件結構作為輸入。但請記住，`Model.fit()`、`Model.evalute()`和 `Model.predict()` 略有不同。由於這些方法強制 `x`（特徵）`y` （標籤或目標）資料作為輸入，`x`和`y`是由後續扮演關鍵字參數角色的物件結構所分離的位置參數。

```javascript
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() 是異步的
`Model.fit()`是用戶在 TensorFlow.js 中進行模型訓練的主要方法。這種方法通常可以長時間運行、持續數秒或分鐘。因此，我們利用 JavaScript 中的 `async` 功能，以便在瀏覽器中運行時不會阻擋 UI 執行緒。這與 JavaScript 中其他潛在的長時間運行的函數相似，例如：`async`[讀取](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)。請注意，`async` 是一種 Python 中不存在的建構函數，雖然 Keras 中的 [fit()](https://keras.io/models/model/#model-class-api) 方法回傳一個 History 物件，但相對地，JavaScript 中的 `fit()` 方法傳回一個 [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) of History，它可以使用 `await()` 方法（如以上範例）或者是使用 `then()` 方法。

## 無 NumPy 的 TensorFlow.js
Python Keras 的用戶經常使用 NumPy 來執行基本的數值和陣列，例如在上面例子中生成的 2D 張量。

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```
在 TensorFLow.js 中，這種基本得數值操作是透過套件本身完成，例如：
```javascript
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```
`tf.*` 命名空間還提供了許多功能給陣列以及線性代數運算如：矩陣乘法，查看更多資訊，請參閱 [TensorFlow.js 核心文件](https://js.tensorflow.org/api/latest/index.html)。
## 使用工廠方法，而不是建構函數
此行是 Python（來自上面的範例） 中建構函數的呼叫：
```python
# Python:
model = keras.Sequential()
```
如果轉換成 JavaScript，等同的建構函數呼叫如下所示：
```javascript
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```
然而我們不使用「新」的建構函數因為 1）新的關鍵字會使的程式更加膨脹 2) 新的建構函數被認為是 JavaScript 當中不良的部分：潛在的缺陷，在 [JavaScript：優良部分](https://www.oreilly.com/) 一書中備受爭論。要在 TensorFlow.js 中創造模型及層，可以使用工廠方法，這些方法皆使用小駝峰式命名法，例如：
```javascript
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```
## 選項字串值是 小駝峰命名法（lowerCamelCase），而不是 蛇形命名法（snake_case）
在 JavaScript 中較常使用駝峰式命名法來命名（例如：參閱[Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)）而 Python 中較常使用蛇形命名法（例如在 Keras 中）。因此我們決定使用小駝峰式命名法來為下列選項的字串值命名：
- 資料格式：例如：`channekFirst` 而非 `channels_first`
- 初始化程式：例如：`glorotNormal` 而非 `glorot_normal`
- 損失和指標：例如：`meanSquaredError` 而非 `mean_squared_error0`，`categoricalCrossentropy` 而非 `categorical_crossentropy`

如上例所示：
```javascript
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

請放心關於模型序列化與反序列化的問題，TensorFlow.js 的內部機制可以確保蛇形命名在 JSON 物件中可以正常運作，例如從 Python Keras 加載準備訓練模型時。

## 使用 apply() 執行 Layer 物件，而不是將它們作為函數使用
在 Keras 中，一個層的物件有 `_call_` 方法來定義。因此使用者可以呼叫該物件來做為函數來使用層的邏輯，例如：
```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```
這個 Python 語法糖藉由 apply（） 方法在 TensorFlow.js 中來執行：
```javascript
// JavaScript:
const myInput = tf.input{shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() 支援對具體張量進行必要的（急切的）評估
目前在 Keras 中，**呼叫**方法只能在（Python） TensorFlow.js 中的 Tensor 物件（假設 為 TensorFLow 後端）上運行，它們是符號性的且不包含實際的數值。這是上一段的範例中所顯示的內容。然而，在 TensorFlow.js 當中層的 apply() 方法可以在符號及命令模式下運行。如果 `apply()` 使用 SymbolicTensor （與 tf.Tensor 相似），傳回的值將會是 SymbolicTensor。這常常在建構模型的時候發生。但是如果 `apply()` 使用實際具體的張量值，將會傳回一個具體的張量。例如：
```javascript
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```
此特色讓人想起（Python）TensorFlow 的 [Eager Execution](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html)。它為模型開發提供了更強大的交互姓及除錯能力，此外還打開了組成動態神經網絡的大門。

## 優化器在 `train` 之下，而非 `optimizers` 之下
在 Keras 中，優化器物件的建構函數會在 `keras.optimizers.` 命名空間下，在 TensorFlow.js 的層中，優化器的工廠方法會在 `tf.train.` 命名空間下。例如：
```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```javascript
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```
## loadModel() 從 URL 加載，而不是 HDF5 文件
在 Keras 中，模型通常都以 HDF5（.h5）文件儲存，且可以在使用 `keras.models.load_model()` 方法進行加載。此方法採用 .h5 的文件路徑。`load_model()` 在 TensorFlow.js 中則是以 `tf.Model()` 表示。由於 HDF5 不是一個友善於瀏覽器的文件格式，因此 `tf.Model()` 需要使用 TensorFlow.js 特定的格式。 `tf.Model()` 將 model.json 文件作為輸入的參數。可以使用 tensorflow.js pip 套件將 Keras HDF5 文件轉換成 model.json
```javascript
// JavaScript:
const model = await tf.loadModel('https://foo.bar/model.json');
```
注意 `tf.loadModel()` 傳回 [tf.Model](https://js.tensorflow.org/api/latest/index.html#class:tf.Model) 的 [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)

一般情況下，在 TensorFLow.js 中 `tf.Model.save` 及 `tf.loadModel` 方法分別用來儲存及加載 `tf.Model` 。我們將這些 API 設計程類似於 Keras 的 [save 及 load_model API](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。但是瀏覽器環境與 Keras 這種主要運行深度學習框架的後端環境完全不同，尤其是在儲存和傳輸路由陣列。因此 TensorFlow.js 與 Keras 中的儲存/加載 APIs 有著一些有趣的差異。更多詳細資訊，請參閱 [Saving and Loading tf Model](https://js.tensorflow.org/tutorials/model-save-load.html) 教學。
