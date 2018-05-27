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
## 使用工廠方法，而不是建構函數
## 選項字串值是 lowerCamelCase，而不是 snake_case
## 使用 apply() 執行 Layer 物件，而不是將它們作為函數使用
## Layer.apply() 支援對具體張量進行必要的（急切的）評估
## loadModel() 從 URL 讀取，而不是 HDF5 文件