tfjs-tutorials 是由社群翻譯的 TensorFlow.js 中文指南。原始資料及圖片來自 [TensorFlow.js Tutorials & Guides](https://js.tensorflow.org/tutorials/)。

## 入門

### [TensorFlow.js 的核心概念](/tutorials/core-concepts.md)

學習 TensorFlow.js 中的核心概念：張量、運算子、模型、層，以及訓練，並學習一些有用的記憶體管理方法，以及如何寫出簡潔（tidy）的程式碼。

### [給 Keras 使用者的 TensorFlow.js 層 API](/tutorials/tfjs-layers-for-keras-users.md)

這份指南解釋了 TensorFlow.js 層 API 和 Keras 的相同和相異之處。

### [如何開始從 X 開始的 TensorFlow.js 的指南](/tutorials/how-to-get-started.md)

這份指南提供了一系列不同領域的 TensorFlow.js 入門資源。

## 訓練模型

### [訓練的第一步：從合成數據中訓練曲線](/tutorials/fit-curve.md)

這份教學示範了從零開始使用 TensorFlow.js 操作建立一個玩具模型。我們會將一個多項式函數調整到適合一些合成數據的曲線。

### [圖像訓練：使用卷積神經網路辨識手寫數字](/tutorials/mnist.md)

這份教學介紹了使用卷積神經網路來辨識圖像（MNIST）中的手寫數字。我們使用 TensorFlow.js 的層 API 來建構、訓練並評估模型。

### [轉移學習：訓練一個神經網路來預測 Webcam 資料](/tutorials/webcam-transfer-learning.md)

這份教學將解釋如何訓練神經網路，以便從 Webcam 資料中進行預測。我們將使用這些預測資料來玩 Pac-Man！

### [轉移學習：建立一個語音辨識模型](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.htm) （外部網站）

在這份教學裡擬將能夠建立一個語音辨識模型，並用它來聲控瀏覽器的聲音滑桿。

### [儲存和載入 tf.Model](/tutorials/model-save-load.md)

這份教學解釋了如何儲存 `tf.Model` 到瀏覽器的 Local Storage 等不同位置，並載入回來。

## 使用預先訓練模型

### [如何將 Keras 模型導入 TensorFlow.js](/tutorials/import-keras.md)

這份教學介紹如何把現有的 Keras 模型轉換到瀏覽器中使用。

### [如何將 TensorFlow SavedModel 導入 TensorFlow.js](https://github.com/tensorflow/tfjs-converter)

**Developer Preview**：這份教學介紹如何把現有的 TensorFlow SavedModel 轉換到瀏覽器中使用。

## 進階項目

### [如何定義自訂的 WebGL 操作](/tutorials/custom-webgo-op.md)（尚未完成）

這份教學如何建立一個自訂的 WebGL 操作，並能用來和 TensorFlow.js 的其他操作一起使用。
