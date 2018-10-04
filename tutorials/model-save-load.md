# 儲存和載入 tf.Model

這份教學描述了 TensorFlow.js 如何儲存並載入模型。儲存和載入模型是一件重要的事情，例如你要怎麼把一個已經被瀏覽器上才能取得的資料（如感測器上的影像、聲音資料）訓練好參數的模型存下來，以供這位使用者下次再瀏覽這個頁面時看到的是已經訓練好的模型？另外也想想層 API 讓你可以在瀏覽器上從頭開始訓練模型（[tf.Model](https://js.tensorflow.org/api/latest/#class:Model)），但怎麼把這個模型存起來？這些問題就得靠儲存和載入 API——從 TensorFlow.js 0.11.1 開始支援的兩個 API。

> 註：這份文件是關於怎麼儲存和載入 `tf.Model`（即 tfjs-layers API 裡的 Keras-style 模型）。儲存和載入 `tf.FrozenModel`（即 TensorFlow SavedModels 載入的模型）目前並不支援，但正在努力讓它支援。

## 儲存 tf.Model

讓我們先從最簡單、最不麻煩的方式來儲存一個 `tf.Model`：存到瀏覽器裡的 Local Storage。Local Storage 是一個標準的客戶端（client-side）儲存空間。在同一個頁面中，資料會被保存以供下次讀取使用。你可以在此 [MDN 頁面](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) 學到更多有關 Local Storage 的訊息。

假設你有個 `tf.Model` 物件叫做 `model`，而他是從頭開始以層 API 或從已訓練好的 Keras 模型載入的話，你可以用這一行程式碼把它存到 Local Storage 裡：

```javascript
const saveResult = await model.save('localstorage://my-model-1');
```

一些值得指出的事情：

- `save` 方法需要一個類似 URL 的字串，它以 **scheme** 開頭。在這個範例中我們使用 `localstorage://` 這個 scheme 來表示這個模型是要被存到 Local Storage 的。
- Scheme 後面會跟著**路徑**。假設是存到 Local Storage 的話，路徑只需要是一個簡單字串，可以唯一識別欲儲存模型即可，供之後使用，例如當你需要載入模型的時候。
- `save` 方法是非同步的，所以如果它是執行其他步驟的先決條件，你需要使用 `then` 或 `await`。
- `model.save` 的回傳值是個帶有一些潛在重要資訊的 JSON 物件，例如模型拓墣和模型參數的位元組大小。
- 任何 `tf.Model`，不管是不是以 [tf.sequential](https://js.tensorflow.org/api/latest/#sequential) 建構而成或裡面包含什麼層，都可以用這個方式儲存。

下面的表格列出了儲存模型支援的所有儲存位置、 scheme 及範例。

儲存位置 | Scheme 字串 | 範例程式碼
-----|------|-----
瀏覽器 Local Storage | `localstorage://` | `await model.save('localstorage://my-model-1');`
瀏覽器 IndexedDB | `indexddb://` | `await model.save('indexddb://my-model-1');`
觸發檔案下載 | `downloads://` | `await model.save('downloads://my-model-1');`
HTTP 請求 | `http://` 或 `https://` | `await model.save('http://model-server.domain/upload');`
檔案系統（Node.js） | `file://` | `await model.save('file:///tmp/my-model-1');`

我們會在接下來的章節解釋其中一些儲存位置。

### IndexedDB

[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) 是另外一個大多數主流網頁瀏覽器都支援的客戶端資料儲存空間。不像 Local Storage，它對大型二進位檔案（BLOBs）有著比較好的支援和更大的儲存空間。因此，把 `tf.Model` 儲存到 IndexedDB 比起 Local Storage 會給你較好的儲存效率及較大的儲存限制。

### 檔案下載

`downloads://` 後面跟的字串是檔案下載時的檔名前綴。例如 `model.save('downloads://my-model-1')` 會讓瀏覽器下載包含同樣前綴的兩個檔案：

1. 一個名為 `my-model-1.json` 的文字 JSON 檔案，裡面包含 `modelTopology`（模型的拓墣）和 `weightsManifest`（權重的表示）兩個欄位。
2. 一個包含權重值的二進位檔案 `my-model-1.weights.bin`。

這兩個檔案的格式和透過 [tensorflowjs 轉換器](https://pypi.org/project/tensorflowjs/) 從 Keras HDF5 轉換過來的檔案格式相同。

註：某些瀏覽器需要先獲得使用者的允許，才能一次下載兩個檔案。

### HTTP 請求

如果 `tf.Model.save` 是和 HTTP/HTTPS URL 一起被呼叫的話，模型的拓墣和權重會被透過 [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) 請求傳送動指定的 HTTP 伺服器，該 POST 請求的是 `multipart/form-data` 格式，這種格式是一個上傳檔案到伺服器用的標準 MIME 格式，裡面則包含兩個檔案：`model.json` 及 `model.weights.bin`，檔案的格式和透過 `downloads://` 觸發檔案下載的格式完全一樣（參考上面的章節）。這份 [文件](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest) 包含一段 Python 代碼，示範如何使用 [flask](http://flask.pocoo.org/) 網頁框架、Keras 和 TensorFlow 來處理原本保存的請求，並將 Keras Model 物件放進伺服器的記憶體中。

在正常情況下，你的 HTTP 伺服器對於請求有特殊的限制和需求，像是 HTTP 方法、表頭和認證用的憑證資訊。你可以透過將 `save` 方法中的 URL 字串換成呼叫 `tf.io.browserHTTPRequest` 對請求的這些方面做細部的控制，這是個比較冗長的 API，但可以在 `save` 提出 HTTP 請求時保持較高的靈活性。例如：

```javascript
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

## 本機檔案系統

TensorFlow.js 可以在 Node.js 環境中使用。有關更多詳細信息，請參閱 [tfjs-node 項目](https://github.com/caisq/tfjs-node)。Node.js 和瀏覽器之間的其中區一個區別是 Node.js 可以直接使用本機檔案系統。因此，`tf.Model` 可以儲存到到檔案系統中。這與在Keras中將模型儲存到硬碟的方式是一樣的。要做到這一點，你必須先導入 `@tensorflow/tfjs-node` 軟件包。例如使用 Node.js 的 `require` 語法：

```javascript
require('@tensorflow/tfjs-node');
```

導入軟件包後，使用 `file://` URL 字串便可將模型儲存和加載。在儲存模型時，URL 字串後面是要儲存模型的目錄的路徑，例如：

```javascript
await model.save('file:///tmp/my-model-1');
```

以上的指令會在 `/tmp/my-model-1` 目錄中生成一個 `model.json` 檔案和一個 `weights.bin` 檔案。這些檔案的格式與[檔案下載](#檔案下載)和[HTTP 請求](#HTTP-請求)部分中所述的格式相同。儲存後的模型可以在導入了 TensorFlow.js 的 Node.js 程序或瀏覽器載入。在 Node.js 環境中，請使用 `tf.loadModel()` 和 `model.json` 檔案的路徑：

```javascript
const model = await tf.loadModel('file:///tmp/my-model-1/model.json');
```

在瀏覽器中，將儲存的檔案作為網絡服務器的靜態文件提供。

## 載入 tf.Model

如果模型不能被讀取，儲存 `tf.Model` 就沒有意義了。透過呼叫 `tf.loadModel`，並傳入一個含有 scheme 的像是的 URL 字串，就能載入模型。大多數的情況這些 URL 字串和 `tf.Model.save` 使用的字串是對稱的，下表總結了目前支持的路由：

儲存位置 | Scheme 字串 | 範例程式碼
-----|------|-----
瀏覽器 Local Storage | `localstorage://` | `await tf.loadModel('localstorage://my-model-1');`
瀏覽器 IndexedDB | `indexddb://` | `await tf.loadModel('indexddb://my-model-1');`
使用者從瀏覽器上傳的檔案 | N/A | `await tf.loadModel(tf.io.browserFiles([modelJSONFile, weightsFile]));`
HTTP 請求 | `http://` 或 `https://` | `await tf.loadModel('http://model-server.domain/download/model.json');`
檔案系統（Node.js） | `file://` | `await tf.loadModel('file:///tmp/my-model-1/model.json');`

在所有載入路由中，如果成功的話，`tf.loadModel` 回傳一個（以 `Promise` 包裝的）`tf.Model` 物件；如果失敗的話，回傳一個 `Error`。

從 Local Storage 或 IndexedDB 載入模型和儲存的方法完全一樣；然而，從使用者上傳的檔案讀取就不完全相同。深入來說，使用者上傳的檔案並不是一個像是 URL 的字串，而是一個包含檔案（[File](https://developer.mozilla.org/en-US/docs/Web/API/File)）物件的陣列（`Array`）。一般來說，工作流程是透過 HTML 檔案上傳（[file input](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file)）元素讓使用者選取本機檔案，例如：

```html
<input name="json-upload" type="file" />
<input name="weights-upload" type="file" />
```

這會在瀏覽器上跑出兩個「瀏覽」按鈕讓使用者選擇檔案。一旦使用者在欄位上選擇了 model.json 檔案和權重檔案，檔案物件就會在對應的 HTML 元素下可用，也就可以像這樣透過 `tf.Models` 來載入：

```javascript
const jsonUpload = document.getElementById('json-upload');
const weightsUpload = document.getElementById('weights-upload');

const model = await tf.loadModel(
    tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
```

從 HTTP 載入一個模型也和透過 HTTP 請求儲存時差不多一樣。特別是，`tf.loadModel` 接受 model.json 檔案的 URL 或路徑，像是上面表格中的範例。這個 API 從第一次釋出 TensorFlow.js 就存在了。

## 管理存在客戶端的模型

像你在上面所學的，你可以把 `tf.Model` 的拓墣和權重存在客戶端的瀏覽器儲存空間，包含 Local Storage 和 IndexedDB，只要透過像是 `model.save('localstorage://my-model')` 和 `model.save('indexeddb://my-model')` 就可以了。不過你要怎麼找出到目前為止有哪些模型已經被儲存？這可以透過 `tf.io` 這個 API 裡面的模型管理方法來完成：

```javascript
// 列出 Local Storage 裡的模型
console.log(await tf.io.listModels());
```

`listModels` 方法的回傳值不只包含儲存模型的路徑，還包含了一些關於模型的元資料，像是模型拓墣和模型參數的位元組大小。

這個管理 API 同時也可以讓你複製、移動或刪除現有模型，例如：

```javascript
// 將現有的模型複製到新的位置。
// 在 Local Storage 和 IndexedDB 之間複製是支援的。
tf.io.copyModel('localstorage://my-model', 'indexeddb://cloned-model');

// 將模型移動到別的位置。
// 在 Local Storage 和 IndexedDB 之間移動是支援的。
tf.io.moveModel('localstorage://my-model', 'indexeddb://cloned-model');

// 移除模型。
tf.io.removeModel('indexeddb://cloned-model');
```

## 將儲存的 `tf.Model` 轉換為 Keras 格式

如上述，將 `tf.Model` 儲存為檔案有兩種方法：

* 在瀏覽器中使用 `downloads://` Scheme 字串下載文件
* 在 Node.js 中使用 `file://` URL 字串將模型直接寫入本機檔案系統。透過 [tensorflowjs 轉換器](https://pypi.org/project/tensorflowjs/) 將那些檔案轉換為 HDF5 格式後 Keras 便可以把它當作 Keras 模型直接使用或讀取，例如：

```Bash
# 假設你已經下載 `my-model-1.json`, 和一個權重檔案。

pip install tensorflowjs

tensorflowjs_converter \
    --input_format tensorflowjs --output_format keras \
    ./my-model-1.json /tmp/my-model-1.h5
```
