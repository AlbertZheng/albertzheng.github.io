let model;
let categoryNames = [];
let canvas;
let drawingCoordinates = [];
let mousePressed = false;
let demoTopK = 3;
let demoMiniCategoryNumber = 10;


(function(doc, win) {
    var docEl = doc.documentElement,
        isIOS = navigator.userAgent.match(/\(i[^;]+;( U;)? CPU.+Mac OS X/),
        dpr = isIOS ? Math.min(win.devicePixelRatio, 3) : 1,
        dpr = window.top === window.self ? dpr : 1, //被iframe引用时，禁止缩放
        dpr = 1,
        scale = 1 / dpr,
        resizeEvt = 'orientationchange' in window ? 'orientationchange' : 'resize';
    docEl.dataset.dpr = dpr;
    var metaEl = doc.createElement('meta');
    metaEl.name = 'viewport';
    metaEl.content = 'initial-scale=' + scale + ',maximum-scale=' + scale + ', minimum-scale=' + scale + ',user-scalable=no';
    docEl.firstElementChild.appendChild(metaEl);
    var recalc = function() {
        var width = docEl.clientWidth;
        if (width / dpr > 750) {
            width = 750 * dpr;
        }
        // 乘以100，px : rem = 100 : 1
        docEl.style.fontSize = 100 * (width / 750) + 'px';
    };
    recalc()
    if (!doc.addEventListener) return;
    win.addEventListener(resizeEvt, recalc, false);
})(document, window);


/**
 * Initial the drawing canvas
 */
$(function () {
  canvas = window._canvas = new fabric.Canvas('canvas');
  canvas.backgroundColor = '#ffffff';
  canvas.isDrawingMode = 0;
  canvas.freeDrawingBrush.color = "black";
  canvas.freeDrawingBrush.width = 10;
  canvas.renderAll();
  canvas.on('mouse:up', function(event) {
    if (canvas.isDrawingMode) {
      performPrediction();
    }
    mousePressed = false
  });
  canvas.on('mouse:down', function(event) {
    mousePressed = true
  });
  canvas.on('mouse:move', function (event) {
    let pointer = canvas.getPointer(event.e);

    if (pointer.x >= 0 && pointer.y >= 0 && mousePressed) {
      drawingCoordinates.push(pointer)
    }
  });
});


/**
 * Load the CNN model
 */
async function appMain() {
  const MODEL_URL = 'https://raw.githubusercontent.com/AlbertZheng/quickdraw-cnn/master/web-model/tensorflowjs_model.pb';
  const WEIGHTS_URL = 'https://raw.githubusercontent.com/AlbertZheng/quickdraw-cnn/master/web-model/weights_manifest.json';

  console.log("### Loading model... ###");
  model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  console.log("### Model loaded. ###");

  console.log("### Predicting... ###");
  const ones = tf.ones([1, 28, 28, 1]);
  model.execute(ones).print(true);

  // Load the category names
  await loadCategories();

  // Now it is enabled to draw on the canvas
  enableDrawing();
}


/**
 * Load the category names
 */
async function loadCategories() {
  let filename = 'mini-categories.txt';

  await $.ajax({
    url: filename,
    dataType: 'text',
  }).done(function(data) {
    categoryNames = data.split(/\n/);

    categoryNames = categoryNames.map(function(KV) {
      return KV.substring(KV.lastIndexOf('=') + 1);
    });

    let category_list = '';
    for (let i = 0; i < demoMiniCategoryNumber; i++) {
      category_list += categoryNames[i];
      if (i < demoMiniCategoryNumber - 1)
        category_list += '，';
    }
    document.getElementById('status').innerHTML = '请画出如下类别之一的图像： <b><p style="margin:0px">'+category_list+'</p></b>';
  });
}


/**
 * Enable drawing on canvas
 */
function enableDrawing() {
  $('button').prop('disabled', false);
  let thickness = document.getElementById('brush-thickness');
  thickness.oninput = function () {
    canvas.freeDrawingBrush.width = this.value;
  };

  canvas.isDrawingMode = 1;
}


/**
 * Erase the canvas
 */
function eraseCanvas() {
  canvas.clear();
  canvas.backgroundColor = '#ffffff';
  drawingCoordinates = [];
}


/**
 * Perform the prediction
 */
function performPrediction() {
  if (drawingCoordinates.length >= 2) {
    const imageData = getImageData();

    // Get the prediction
    const y_output = model.execute(distort(imageData));
    const probabilities = y_output.dataSync();

    console.log("Model output tensor: ");
    y_output.print(true);
    console.log("Probabilities: ", probabilities);

    // Map the probabilities to indices
    const indices = probabilities.slice(0).sort(function (a, b) {
      return b - a
    }).map(function (probability) {
      for (let i = 0; i < probabilities.length; i++) {
        if (probability === probabilities[i]) {
          return i;
        }
      }
    });

    let topK = (indices.length > demoTopK ? demoTopK : indices.length);
    let predictionText = '';
    for (let i = 0; i < topK; i++) {
      let index = indices[i];
      predictionText += categoryNames[index];
      predictionText += '<span style="font-size:0.22rem;">(匹配度';
      predictionText += (probabilities[index]*100).toFixed(2);
      predictionText += '%)</span>';
      if (i < demoTopK - 1)
        predictionText += ' > ';
    }

    document.getElementById('prediction-result').innerHTML = predictionText;
  }
}


/**
 * Get the image data of current drawing
 */
function getImageData() {
  // Get the border box around the drawing
  const box = getBorderBox();

  // Get the image data according to DPI
  const dpi = window.devicePixelRatio;
  return canvas.contextContainer.getImageData(box.topLeft.x * dpi, box.topLeft.y * dpi,
      (box.bottomRight.x - box.topLeft.x + 1) * dpi, (box.bottomRight.y - box.topLeft.y + 1) * dpi);
}


/**
 * Get the border box
 */
function getBorderBox() {
  let coordinateX = drawingCoordinates.map(function (pointer) {
    return pointer.x
  });
  let coordinateY = drawingCoordinates.map(function (pointer) {
    return pointer.y
  });

  // Get the (top, left) and (bottom, right) points.
  let topLeftCoordinate = {
    x: Math.min.apply(null, coordinateX),
    y: Math.min.apply(null, coordinateY)
  };

  let bottomRightCoordinate = {
    x: Math.max.apply(null, coordinateX),
    y: Math.max.apply(null, coordinateY)
  };

  return {
    topLeft: topLeftCoordinate,
    bottomRight: bottomRightCoordinate
  }
}


/**
 * Distort the drawing data
 */
function distort(imageData) {
  return tf.tidy(function() {
    // The shape is (h, w, 1)
    let tensor = tf.fromPixels(imageData, numChannels = 1);

    // Resize to 28x28 and normalize to 0 (black) and 1 (white)
    const resizedImage = tf.image.resizeBilinear(tensor, [28, 28]).toFloat();
    const normalizedImage = tf.ceil(resizedImage.div(tf.scalar(255.0)));

    // Add a dimension to get a batch shape so that the shape will become (1, h, w, 1)
    return normalizedImage.expandDims(0);
  })
}
