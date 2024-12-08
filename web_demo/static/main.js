const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const audioSource = document.getElementById('audioSource');
const audio = document.getElementById('audio');

// 缩放到128x128
const resizedCanvas = document.createElement('canvas');
const resizedCtx = resizedCanvas.getContext('2d');

let dataSets = [];
let currentDataSetIndex = 0;


let lastTime = performance.now();
const FPS_AVERAGING_WINDOW = 100; // Average the FPS over the last 100 frames
async function init() {
    video.addEventListener('loadedmetadata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.addEventListener('play', processVideoFrames);
    });
}

async function processImage(imageData, points) {
  const { width, height, data } = imageData;

  // 将 ImageData 转换为 ArrayBuffer
  const arrayBuffer = data.buffer;

  // 创建一个视图来操作 ArrayBuffer
  const view = new Uint8Array(arrayBuffer);

  // 将 points 转换为 Float32Array 并调整为字节数组
  const pointsArray = new Float32Array(points);
  const pointsBytes = new Uint8Array(pointsArray.buffer);

  // 计算总长度：图像数据 + 点数据
  const totalLength = arrayBuffer.byteLength + pointsBytes.byteLength;

  // 分配 WebAssembly 堆上的内存，并将数据复制过去
  const combinedPtr = Module._malloc(totalLength);

  // 复制图像数据
  Module.HEAPU8.set(view, combinedPtr);

  // 计算点数据的起始位置
  const pointsOffset = arrayBuffer.byteLength;

  // 复制点数据
  Module.HEAPU8.set(pointsBytes, combinedPtr + pointsOffset);

  // 调用 WebAssembly 函数并传递图像数据指针、图像数据大小、点数据大小
  Module._process_image(combinedPtr, arrayBuffer.byteLength, pointsBytes.byteLength);

  // 获取处理后的图像数据
  const processedView = new Uint8ClampedArray(Module.HEAPU8.buffer, combinedPtr, arrayBuffer.byteLength);
//  console.log("processedView", processedView.length, arrayBuffer.byteLength, width, height)
  // 将处理后的数据放入 ImageData 对象中
  const processedImageData = new ImageData(processedView, width, height); // 使用原始宽度高度

  // 更新 canvas 上的图像显示
  resizedCtx.putImageData(processedImageData, 0, 0);

  // 释放分配的内存
  Module._free(combinedPtr);
}

// 加载JSON数据
async function loadJsonData() {
  try {
    const response = await fetch('json_data.json');
    if (!response.ok) {
      throw new Error('Network response was not ok ' + response.statusText);
    }
    dataSets = await response.json();
    console.log('Data loaded successfully:', dataSets.length, 'sets.');
  } catch (error) {
    console.error('Error loading the file:', error);
  }
}

loadJsonData();

async function processVideoFrames() {

  audio.play();
  let lastDataSetIndex = -1; // 初始化为一个不可能的索引值
  let isProcessing = false; // 标志位

  let lastVideoTime = 0; // 初始化为一个不可能的索引值

  const frameCallback = async (currentTime) => {
    if (!video.paused && !video.ended && !isProcessing) {
      isProcessing = true;

      try {
        // 计算当前数据集索引
        const currentDataSetIndex = Math.floor(video.currentTime * 25);

//        console.log("currentDataSetIndex", currentDataSetIndex, video.currentTime, video.currentTime - lastVideoTime)
        lastVideoTime = video.currentTime
        if (lastDataSetIndex !== currentDataSetIndex && currentDataSetIndex < dataSets.length - 1) {
          lastDataSetIndex = currentDataSetIndex;

          // 清除画布并绘制当前视频帧到canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // 处理当前数据集
          if (currentDataSetIndex < dataSets.length - 1) {
            const dataSet = dataSets[currentDataSetIndex];
            const rect = dataSet.rect;

            const currentTimeStamp = 0.04 * currentDataSetIndex;
            const nextTimeStamp = 0.04 * (currentDataSetIndex + 1);
            const currentpoints = dataSets[currentDataSetIndex].points;
            const nextpoints = dataSets[currentDataSetIndex + 1].points;

            // 线性插值计算
            const t = (video.currentTime - currentTimeStamp) / (nextTimeStamp - currentTimeStamp);
            let points = currentpoints.map((xi, index) => (1-t) * xi + t * nextpoints[index]);

            // 创建临时画布用于裁剪、缩放和绘点
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');

            // 获取rect区域图像数据并绘制到临时画布
            tempCanvas.width = rect[2] - rect[0];
            tempCanvas.height = rect[3] - rect[1];
            tempCtx.drawImage(
              video,
              rect[0],
              rect[1],
              rect[2] - rect[0],
              rect[3] - rect[1],
              0,
              0,
              tempCanvas.width,
              tempCanvas.height
            );

            // 缩放到128x128
            resizedCanvas.width = 128;
            resizedCanvas.height = 128;
            resizedCtx.drawImage(tempCanvas, 0, 0, 128, 128);
            
//            // 在128x128图像上按照points来进行画点，points是二维点坐标列表
//            let subPoints = points.slice(16);
//            // 遍历子列表，每两个元素作为一个坐标对进行处理
//            for (let i = 0; i < subPoints.length; i += 2) {
//              if (i + 1 < subPoints.length) { // 确保有配对的y值
//                let x = subPoints[i];
//                let y = subPoints[i + 1];
//                // 绘制点
//                resizedCtx.beginPath();
//                resizedCtx.arc(x, y, 2, 0, 2 * Math.PI);
//                resizedCtx.fillStyle = 'red'; // 点的颜色
//                resizedCtx.fill();
//                resizedCtx.closePath();
//              }
//            }
            // 获取128x128图像数据并处理
            const imageData = resizedCtx.getImageData(0, 0, 128, 128);
            await processImage(imageData, points);
            // 恢复图像到原始尺寸
            tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(resizedCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

            // 将临时画布的内容放回原画布
            ctx.drawImage(tempCanvas, rect[0], rect[1]);
          }
        }

        isProcessing = false; // 处理完成后将标志位置为false
      } catch (error) {
        console.error('Error processing frame:', error);
        isProcessing = false; // 即使出错也要将标志位置为false
      }

      requestAnimationFrame(frameCallback);
    }
  };

  requestAnimationFrame(frameCallback);
}


init();

document.getElementById('uploadButton').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        readWavFile(file);
    }
});

function readWavFile(file) {
    const reader = new FileReader();

    reader.onload = function(event) {
        const arrayBuffer = event.target.result;
        const dataView = new DataView(arrayBuffer);

        // Check if the file is a valid WAV file
        if (dataView.getUint32(0, true) !== 0x46464952 || dataView.getUint32(8, true) !== 0x45564157) {
            alert('Not a valid WAV file');
            return;
        }

        // Get the PCM data chunk
        const chunkSize = dataView.getUint32(4, true);
        const format = dataView.getUint32(12, true);
        const subChunk1Size = dataView.getUint32(16, true);
        const audioFormat = dataView.getUint16(20, true);
        const numChannels = dataView.getUint16(22, true);
        const sampleRate = dataView.getUint32(24, true);
        const byteRate = dataView.getUint32(28, true);
        const blockAlign = dataView.getUint16(32, true);
        const bitsPerSample = dataView.getUint16(34, true);

        // Find the data chunk
        let dataOffset = 36;
        while (dataOffset < arrayBuffer.byteLength) {
            const chunkId = dataView.getUint32(dataOffset, true);
            const chunkSize = dataView.getUint32(dataOffset + 4, true);
            if (chunkId === 0x61746164) { // "data" chunk
                const data = new Uint16Array(arrayBuffer, dataOffset + 8, chunkSize / 2);
                console.log('PCM Data:', data);
                // Convert PCM data to Uint8Array
                const view = new Uint8Array(arrayBuffer);

                // Allocate memory in WebAssembly heap
                const arrayBufferPtr = Module._malloc(arrayBuffer.byteLength);

                // Copy data to WebAssembly heap
                Module.HEAPU8.set(view, arrayBufferPtr);

                // Call WebAssembly module's C function
                console.log("buffer.byteLength", arrayBuffer.byteLength);
                Module._setAudioBuffer(arrayBufferPtr, arrayBuffer.byteLength);

                // Free the allocated memory
                Module._free(arrayBufferPtr);

                // Set the audio source and play it when the video starts

                audioSource.src = URL.createObjectURL(new Blob([arrayBuffer], { type: 'audio/wav' }));
                audio.load();

                break;
            }
            dataOffset += 8 + chunkSize;
        }
    };

    reader.readAsArrayBuffer(file);
}