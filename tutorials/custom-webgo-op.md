# 建立自訂的 WebGL 操作

要定義 WebGL 的操作的話，我們要做的事情就是創造一個實作 `tf.webgl.GPGPUProgram` 的物件。

這個介面的定義如下：

```ts
interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting?: boolean;
}
```

對於一個更人為的例子，讓我們來實作一個運算 `f(x) = x * x + x` 的操作。

這個 GLSL 的程式碼將會是這樣子：

```glsl
void main() {
    float x = getXAtOutCoords();
    float value = x * x + x;
    setOutput(value);
}
```

`getXAtOutCoords` 跟 `setOutput` 這兩個東西是 [由 Tensorflow.js 提供](https://js.tensorflow.org/tutorials/custom-webgl-op.html#stdlib) 給著色器的，注意到主函式是被每個輸出張量裡面的值呼叫。

完整的 GPGPUPorgram 定義如下：

```javascript
const squareAndAddKernel = inputShape => ({
  variableNames: ['X'],
  outputShape: inputShape.slice(),
  userCode: `
    void main() {
        float x = getXAtOutCoords();
        float value = x * x + x;
        setOutput(value);
      }
  `
})
```

要執行這項操作，你要使用 `tf.ENV.backend.compileAndRun(program: GPGPUProgram, inputs: tf.Tensor[]): tf.Tensor`，請注意：如果後端不是 webgl，這將會是 undefined。

```javascript
const x = tf.tensor([1, 2, 3, 4]);
const program = squareAndAddKernel(x.shape);

const result = tf.ENV.backend.compileAndRun(program, [x]);
```

然而，我們可能也想又定義這項操作的梯度，這樣一來梯度就可以透過這項操作被反向傳播。

為了做到這件事情，我們使用 [tf.costomGrad](https://js.tensorflow.org/api/latest/#customGrad)。

```javascript
const squareAndAddBackpropKernel = inputShape => ({
  variableNames: ['X'],
  outputShape: inputShape.slice(),
  userCode: `
    void main() {
      float x = getXAtOutCoords();
      float value = 2.0 * x + 1.0;
      setOutput(value);
    }
  `
});


const squareAndAdd = tf.customGrad(x => {
  const backend = tf.ENV.backend;
  const program = squareAndAddKernel(x.shape);
  const backpropProgram = squareAndAddBackpropKernel(x.shape);

  const value = backend.compileAndRun(program, [x]);

  const gradFunc = dy =>
      [backend.compileAndRun(backpropProgram, [x]).mul(dy)];
  return {value, gradFunc};
});
```

我們也可以這樣使用：

```javascript
const x = tf.tensor([1, 2, 3, 4]);

const value = squareAndAdd(x);

const grads = tf.grad(x => squareAndAdd(x));
const dx = grads(x);

// value == [2, 6, 12, 20]
// dx == [3, 5, 7, 9]
```

或是更簡明的：

```javascript
const {value, grad} = tf.valueAndGrad(squareAndAdd)(x);
```

# 由 Tensorflow.js 產生的 GLSL 函式

Tensorflow.js 產生讓你可以使用來從張量輸出以及寫入輸出張量的函式，還有其他數字相關的有用函式，這些函式都是由 [Shader Compiler](https://github.com/tensorflow/tfjs-core/blob/master/src/kernels/webgl/shader_compiler.ts) 添加進你的程式碼裡面。

* `void setOutput(float value)`
  * 設定片段著色運行的座標輸出（相當於 `gl_FragCoord = vec4(value, 0.0, 0.0, 0.0)`）
* `indexType getOutputCoords()`
  * `indexType` 就是 `int | ivec2 | ivec3 | ivec4 | ivec5 | ivec6` 其中的一個。
  * 如果輸出張量是 rank-0 或是 rank-1 的話，會回傳 `int`，否則將會回傳 `iveN`，rank 是多少 N 就是多少，這是這個程式將會寫入的在輸出張量裡面的單元的座標。
* Tensorflow.js 產生 GLSL 函式去從輸入張量中取樣，這些是它們的格式：

```javascript
float get{VarName}AtOutCoords()

float get{VarName}() // rank-0 input
float get{VarName}(int x) // rank-1 input
float get{VarName}(int x, int y) // rank-2 input
float get{VarName}(int x, int y, int z) // rank-3 input
float get{VarName}(int x, int y, int z, int w) // rank-4 input
// continue as above for rank-5 & rank-6

// For example, for rank-2 Tensor named x:
// float getX(int x, int y)
```

`VarName` 是一個**第一個字母是大寫**被定義在 `GPGPUProgram` 的 `varialbleNames` 陣列的變數名稱。這意味著對於一個名稱為 `matrix` 的變數，TF.js 將會產生 `getMatrix` 名叫的變數。

許多這些函式是依賴輸入張量的 rank，所以在你的 `GPGPUProgram` 中，你將會常常想要觸發不同的基於 `inputshape`s 的排名的程式碼，比如說，如果`get{VarName}AtOutCoords()` 不存在的話，我們也許會將 `squareAndAddKernel` 寫成：

```javascript
const squareAndAddKernel = inputShape => ({
  const variableNames = ['X']
  const outputShape = inputShape.slice()
  const rank = outputShape.length

  const coordSnippets = ['',
      'coords',
      'coords.x, coords.y',
      'coords.x, coords.y, coords.z',
      'coords.x, coords.y, coords.z, coords.w']

  const coordType = rank < 2 ? 'int' : `ivec${rank}`

  const userCode = `
    void main() {
      ${coordType} coords = getOutputCoords();
      float x = getX(${coordSnippets[rank]});
      setOutput(x * x + x);
    }`

  return {variableNames, outputShape, userCode}
})
```

  * `bool isNaN(float val)`
    * 如果 val 為 `NaN` 的話為 `true`，否則為 false。
  * `int round(float value)`
    * 將 `vaule` 四捨五入到最近的整數。
  * `int imod(int x, int y)`
    * 如同 `float mod(float x, float y)` 但是是針對整數的，因為 GPGL 並沒有提供相關函式。
  * `float random(float seed)`
    * 回傳一個基於 [Dave Hoskins 方程式](https://www.shadertoy.com/view/4djSRW) 的偽隨機數字。
