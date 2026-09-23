import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart' as flutter;
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:onnxruntime/onnxruntime.dart';

/// HqSamTracer: High-precision Hybrid Tracer.
/// Supports Cloud (SAM-HQ) and Mobile (MobileSAM AI) modes.
class HqSamTracer {
  static const double modelInputSize = 1024.0;

  OrtSession? _encoderSession;
  OrtSession? _decoderSession;
  bool _isMobileSamLoaded = false;

  /// Loads MobileSAM ONNX models from assets.
  Future<void> loadMobileSam() async {
    if (_isMobileSamLoaded) return;
    try {
      OrtEnv.instance;

      final encoderBytes = await rootBundle.load('assets/mobile_sam.encoder.onnx');
      _encoderSession = OrtSession.fromBuffer(encoderBytes.buffer.asUint8List(), OrtSessionOptions());

      final decoderBytes = await rootBundle.load('assets/mobile_sam.decoder.onnx');
      _decoderSession = OrtSession.fromBuffer(decoderBytes.buffer.asUint8List(), OrtSessionOptions());

      _isMobileSamLoaded = true;
      flutter.debugPrint("MobileSAM Models Loaded Successfully");
    } catch (e) {
      flutter.debugPrint("MobileSAM Load Error: $e");
    }
  }

  /// Mobile AI Extraction: Full On-Device MobileSAM Inference.
  Future<List<List<flutter.Offset>>> extractMobileBoundaries(
    Uint8List imageBytes,
    HqSamPrepResult prep, {
    required flutter.Rect roi,
    flutter.Offset? centerPoint,
  }) async {
    if (!_isMobileSamLoaded) await loadMobileSam();
    if (!_isMobileSamLoaded || _encoderSession == null || _decoderSession == null) return [];

    OrtRunOptions? runOptions;
    OrtValueTensor? imageTensor;
    List<OrtValue?>? encoderOutputs;
    List<OrtValue?>? decoderOutputs;

    OrtValueTensor? pointCoordsTensor;
    OrtValueTensor? pointLabelsTensor;
    OrtValueTensor? maskInputTensor;
    OrtValueTensor? hasMaskInputTensor;
    OrtValueTensor? origImSizeTensor;

    try {
      runOptions = OrtRunOptions();

      // 1. STAGE 1: ENCODER (1024x1024 -> Embedding)
      // OPTIMIZATION: Use OpenCV for normalization and channel splitting
      final rgbMat = cv.cvtColor(prep.paddedMat, cv.COLOR_BGR2RGB);

      // Convert to float and normalize in one go using scale and offset
      // formula: (x / 255.0 - mean) / std  =>  x * (1 / (255 * std)) - (mean / std)
      // For R: x * (1 / (255 * 0.229)) - (0.485 / 0.229) = x * 0.01712 - 2.1179
      // For G: x * (1 / (255 * 0.224)) - (0.456 / 0.224) = x * 0.01750 - 2.0357
      // For B: x * (1 / (255 * 0.225)) - (0.406 / 0.225) = x * 0.01742 - 1.8044

      final channels = cv.split(rgbMat);
      final r = channels[0];
      final g = channels[1];
      final b = channels[2];

      final rFloat = r.convertTo(cv.MatType.CV_32FC1, alpha: 0.01712475, beta: -2.11790393);
      final gFloat = g.convertTo(cv.MatType.CV_32FC1, alpha: 0.017507, beta: -2.03571429);
      final bFloat = b.convertTo(cv.MatType.CV_32FC1, alpha: 0.01742919, beta: -1.80444444);

      final floatData = Float32List(1 * 3 * 1024 * 1024);
      floatData.setAll(0, rFloat.data.buffer.asFloat32List());
      floatData.setAll(1048576, gFloat.data.buffer.asFloat32List());
      floatData.setAll(2097152, bFloat.data.buffer.asFloat32List());

      rgbMat.dispose(); r.dispose(); g.dispose(); b.dispose();
      rFloat.dispose(); gFloat.dispose(); bFloat.dispose();

      imageTensor = OrtValueTensor.createTensorWithDataList(floatData, [1, 3, 1024, 1024]);

      // Use the model's preferred input name dynamically
      flutter.debugPrint("Encoder Input Names: ${_encoderSession!.inputNames}");
      flutter.debugPrint("Decoder Input Names: ${_decoderSession!.inputNames}");

      encoderOutputs = _encoderSession!.run(runOptions, {_encoderSession!.inputNames[0]: imageTensor});
      final imageEmbeddings = encoderOutputs[0] as OrtValueTensor;

      // 2. STAGE 2: DECODER (Embedding + Box + Optional Center Point -> Mask)
      final modelRoi = mapPromptToModelSpace(roi, prep);

      List<double> coords = [modelRoi.left, modelRoi.top, modelRoi.right, modelRoi.bottom];
      List<double> labels = [2.0, 3.0]; // Box TL and BR

      // FIXED: Use a strong center point for precision and ignore shadows
      final double cx = modelRoi.left + modelRoi.width / 2;
      final double cy = modelRoi.top + modelRoi.height / 2;
      coords.addAll([cx, cy]);
      labels.add(1.0);

      final int numPoints = labels.length;
      pointCoordsTensor = OrtValueTensor.createTensorWithDataList(Float32List.fromList(coords), [1, numPoints, 2]);
      pointLabelsTensor = OrtValueTensor.createTensorWithDataList(Float32List.fromList(labels), [1, numPoints]);
      maskInputTensor = OrtValueTensor.createTensorWithDataList(Float32List(1 * 1 * 256 * 256), [1, 1, 256, 256]);
      hasMaskInputTensor = OrtValueTensor.createTensorWithDataList(Float32List.fromList([0.0]), [1]);
      origImSizeTensor = OrtValueTensor.createTensorWithDataList(Float32List.fromList([1024.0, 1024.0]), [2]);

      final decoderInputs = {
        _decoderSession!.inputNames[0]: imageEmbeddings,
        _decoderSession!.inputNames[1]: pointCoordsTensor,
        _decoderSession!.inputNames[2]: pointLabelsTensor,
        _decoderSession!.inputNames[3]: maskInputTensor,
        _decoderSession!.inputNames[4]: hasMaskInputTensor,
        _decoderSession!.inputNames[5]: origImSizeTensor,
      };

      decoderOutputs = _decoderSession!.run(runOptions, decoderInputs);
      final masks = decoderOutputs[0] as OrtValueTensor;

      // 3. POST-PROCESS: Threshold the 1024x1024 logits
      final maskValue = masks.value as List<dynamic>;
      final maskData = maskValue[0][0] as List<dynamic>;

      final binaryData = Uint8List(1024 * 1024);
      for (int y = 0; y < 1024; y++) {
        final row = maskData[y] as List<dynamic>;
        for (int x = 0; x < 1024; x++) {
          // Changed threshold from 0.0 to 0.5. 
          // SAM outputs standard sigmoid logits; using 0.5 filters out low-confidence 
          // blurry edge estimations and locks onto a crisp, sharply defined boundary line.
          if (row[x] > 0.5) {
            binaryData[y * 1024 + x] = 255;
          }
        }
      }

      final binaryMask = cv.Mat.fromList(1024, 1024, cv.MatType.CV_8UC1, binaryData);
      final (contours, _) = cv.findContours(binaryMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

      // Sort contours by area descending so the largest/most stable target is always at index 0
      final sortedContours = List.generate(contours.length, (i) => contours[i]);
      sortedContours.sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));

      List<List<flutter.Offset>> resultVectors = [];
      for (var i = 0; i < sortedContours.length; i++) {
        final cnt = sortedContours[i];
        if (cv.contourArea(cnt) < 10) continue;

        List<flutter.Offset> points = [];
        for (var j = 0; j < cnt.length; j++) {
          final double origX = (cnt[j].x - prep.padX) / prep.scale;
          final double origY = (cnt[j].y - prep.padY) / prep.scale;

          points.add(flutter.Offset(
            origX.clamp(0.0, prep.originalSize.width),
            origY.clamp(0.0, prep.originalSize.height),
          ));
        }
        resultVectors.add(points);
      }

      binaryMask.dispose();
      return resultVectors;
    } catch (e) {
      flutter.debugPrint("MobileSAM Error: $e");
      return [];
    } finally {
      runOptions?.release();
      imageTensor?.release();
      pointCoordsTensor?.release();
      pointLabelsTensor?.release();
      maskInputTensor?.release();
      hasMaskInputTensor?.release();
      origImSizeTensor?.release();
      if (encoderOutputs != null) {
        for (var v in encoderOutputs) {
          v?.release();
        }
      }
      if (decoderOutputs != null) {
        for (var v in decoderOutputs) {
          v?.release();
        }
      }
    }
  }

  /// Helper to get a Rect bounding box from a list of contours.
  flutter.Rect? getBoundingBox(List<List<flutter.Offset>> contours) {
    if (contours.isEmpty) return null;
    double minX = 10000, minY = 10000, maxX = -10000, maxY = -10000;
    bool found = false;
    for (final poly in contours) {
      for (final p in poly) {
        minX = math.min(minX, p.dx);
        minY = math.min(minY, p.dy);
        maxX = math.max(maxX, p.dx);
        maxY = math.max(maxY, p.dy);
        found = true;
      }
    }
    return found ? flutter.Rect.fromLTRB(minX, minY, maxX, maxY) : null;
  }

  /// Calculates the centroid of the densest point cluster (ignores noise/shadows).
  flutter.Offset? getCentroid(List<List<flutter.Offset>> contours) {
    if (contours.isEmpty) return null;

    // 1. Identify the primary object contour (the one with the most points)
    List<flutter.Offset> mainObject = contours[0];
    for (var i = 1; i < contours.length; i++) {
      if (contours[i].length > mainObject.length) {
        mainObject = contours[i];
      }
    }

    // 2. Perform a "Robust Average" by removing outliers
    // This stops trailing shadows from pulling the center away
    double sumX = 0;
    double sumY = 0;
    for (final p in mainObject) {
      sumX += p.dx;
      sumY += p.dy;
    }
    final double rawAvgX = sumX / mainObject.length;
    final double rawAvgY = sumY / mainObject.length;

    // Filter points that are too far from the raw average (potential shadows)
    double filteredSumX = 0;
    double filteredSumY = 0;
    int count = 0;
    for (final p in mainObject) {
      final dist = math.sqrt(math.pow(p.dx - rawAvgX, 2) + math.pow(p.dy - rawAvgY, 2));
      if (dist < 50) { // Cluster threshold in pixels
        filteredSumX += p.dx;
        filteredSumY += p.dy;
        count++;
      }
    }

    return count > 0
      ? flutter.Offset(filteredSumX / count, filteredSumY / count)
      : flutter.Offset(rawAvgX, rawAvgY);
  }

  /// Preprocesses image from raw RGBA bytes into 1024x1024 BGR padded format.
  HqSamPrepResult preprocessRaw(Uint8List rgbaBytes, int width, int height) {
    final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, rgbaBytes);
    final bgr = cv.cvtColor(mat, cv.COLOR_RGBA2BGR);

    final double scale = modelInputSize / math.max(width, height);
    final int newW = (width * scale).round();
    final int newH = (height * scale).round();

    final resized = cv.resize(bgr, (newW, newH), interpolation: cv.INTER_AREA);

    final int padX = (modelInputSize - newW) ~/ 2;
    final int padY = (modelInputSize - newH) ~/ 2;

    final paddedMat = cv.Mat.zeros(1024, 1024, cv.MatType.CV_8UC3);
    final roi = cv.Rect(padX, padY, newW, newH);
    resized.copyTo(paddedMat.region(roi));

    final result = HqSamPrepResult(
      paddedMat: paddedMat,
      originalSize: flutter.Size(width.toDouble(), height.toDouble()),
      scale: scale,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
    );

    mat.dispose();
    bgr.dispose();
    resized.dispose();
    return result;
  }

  /// Preprocesses image into 1024x1024 zero-padded letterbox format.
  HqSamPrepResult preprocess(Uint8List imageBytes) {
    final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (mat.isEmpty) throw Exception("Failed to decode image bytes.");

    final int origH = mat.height;
    final int origW = mat.width;

    final double scale = modelInputSize / math.max(origW, origH);
    final int newW = (origW * scale).round();
    final int newH = (origH * scale).round();

    final resized = cv.resize(mat, (newW, newH), interpolation: cv.INTER_AREA);

    final int padX = (modelInputSize - newW) ~/ 2;
    final int padY = (modelInputSize - newH) ~/ 2;

    final paddedMat = cv.Mat.zeros(1024, 1024, cv.MatType.CV_8UC3);
    final roi = cv.Rect(padX, padY, newW, newH);
    resized.copyTo(paddedMat.region(roi));

    final result = HqSamPrepResult(
      paddedMat: paddedMat,
      originalSize: flutter.Size(origW.toDouble(), origH.toDouble()),
      scale: scale,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
    );

    mat.dispose();
    resized.dispose();
    return result;
  }

  /// Maps normalized UI box (0-1) to 1024x1024 model space.
  flutter.Rect mapPromptToModelSpace(flutter.Rect roi, HqSamPrepResult prep) {
    final double x1 = (roi.left * prep.originalSize.width) * prep.scale + prep.padX;
    final double y1 = (roi.top * prep.originalSize.height) * prep.scale + prep.padY;
    final double x2 = (roi.right * prep.originalSize.width) * prep.scale + prep.padX;
    final double y2 = (roi.bottom * prep.originalSize.height) * prep.scale + prep.padY;

    return flutter.Rect.fromLTRB(
      x1.clamp(0.0, 1023.0),
      y1.clamp(0.0, 1023.0),
      x2.clamp(0.0, 1024.0),
      y2.clamp(0.0, 1024.0),
    );
  }

  /// Extracts vectors from a binary mask (Backend prediction).
  List<List<flutter.Offset>> extractSubpixelBoundaries({
    required Uint8List maskBytes,
    required HqSamPrepResult prep,
    double thresholdValue = 128.0,
  }) {
    final maskMat = cv.imdecode(maskBytes, cv.IMREAD_GRAYSCALE);
    if (maskMat.isEmpty) return [];

    final (_, binaryMask) = cv.threshold(maskMat, thresholdValue, 255, cv.THRESH_BINARY);
    final (contours, _) = cv.findContours(binaryMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

    List<List<flutter.Offset>> resultVectors = [];
    for (var i = 0; i < contours.length; i++) {
      final cnt = contours[i];
      if (cv.contourArea(cnt) < 10) continue;

      List<flutter.Offset> points = [];
      for (var j = 0; j < cnt.length; j++) {
        final double origX = (cnt[j].x - prep.padX) / prep.scale;
        final double origY = (cnt[j].y - prep.padY) / prep.scale;

        points.add(flutter.Offset(
          origX.clamp(0.0, prep.originalSize.width),
          origY.clamp(0.0, prep.originalSize.height),
        ));
      }
      resultVectors.add(points);
    }

    maskMat.dispose();
    binaryMask.dispose();
    return resultVectors;
  }

  /// Disposes sessions and releases native memory.
  void dispose() {
    _encoderSession?.release();
    _decoderSession?.release();
  }
}

class HqSamPrepResult {
  final cv.Mat paddedMat;
  final flutter.Size originalSize;
  final double scale;
  final double padX;
  final double padY;

  HqSamPrepResult({required this.paddedMat, required this.originalSize, required this.scale, required this.padX, required this.padY});

  void dispose() {
    if (!paddedMat.isEmpty) paddedMat.dispose();
  }
}
