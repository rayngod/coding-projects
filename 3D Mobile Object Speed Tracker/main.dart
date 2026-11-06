import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:io';
import 'dart:math';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/rendering.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:onnxruntime/onnxruntime.dart';

// =============================================================================
// APP ENTRY & CAMERA RECORDING SCREEN
// =============================================================================

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  MediaKit.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Speed Tracker',
      theme: ThemeData.dark(),
      home: const CameraFrameScreen(),
    );
  }
}

class CameraFrameScreen extends StatefulWidget {
  const CameraFrameScreen({super.key});
  @override
  State<CameraFrameScreen> createState() => _CameraFrameScreenState();
}

class _CameraFrameScreenState extends State<CameraFrameScreen> {
  CameraController? _controller;
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();
    if (_cameras.isNotEmpty) {
      _controller = CameraController(_cameras[0], ResolutionPreset.high);
      _controller!.initialize().then((_) {
        if (mounted) setState(() {});
      });
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final file = await _controller!.stopVideoRecording();
      setState(() => _isRecording = false);
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => VideoPlayerScreen(videoFile: File(file.path), isRecordedInApp: true),
          ),
        );
      }
    } else {
      await _controller!.startVideoRecording();
      setState(() => _isRecording = true);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) return const Scaffold();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Record or Upload Video'),
        actions: [
          IconButton(
            icon: const Icon(Icons.photo_library),
            onPressed: () async {
              final video = await ImagePicker().pickVideo(source: ImageSource.gallery);
              if (video != null && context.mounted) {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => VideoPlayerScreen(videoFile: File(video.path), isRecordedInApp: false),
                  ),
                );
              }
            },
          ),
        ],
      ),
      body: Stack(
        alignment: Alignment.bottomCenter,
        children: [
          CameraPreview(_controller!),
          Padding(
            padding: const EdgeInsets.all(20),
            child: FloatingActionButton(
              backgroundColor: _isRecording ? Colors.red : Colors.white,
              onPressed: _toggleRecording,
              child: Icon(
                _isRecording ? Icons.stop : Icons.videocam,
                color: _isRecording ? Colors.white : Colors.red,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// VIDEO PLAYER & TRACKER SCREEN
// =============================================================================

class VideoPlayerScreen extends StatefulWidget {
  final File videoFile;
  final bool isRecordedInApp;
  const VideoPlayerScreen({
    super.key,
    required this.videoFile,
    this.isRecordedInApp = false,
  });
  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late final Player _player = Player();
  late final VideoController _controller = VideoController(_player);

  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isSelectionMode = true;
  bool _showGraphs = false;
  bool _stopTrackingRequested = false;
  bool _isSeeking = false;
  bool _seekInFlight = false;

  Rect? _rect;
  List<Offset> _boundaries = [];

  final TextEditingController _h = TextEditingController(text: "1.0");
  final TextEditingController _fMM = TextEditingController(text: "26.0");
  final TextEditingController _samInterval = TextEditingController(text: "1");
  String _calculationResult = "Define range, draw box, then track.";

  double _processingProgress = 0.0;
  int? _startTrackMs;
  int? _endTrackMs;
  Map<int, Rect> _trackedRects = {};
  Map<int, List<Offset>> _trackedBoundaries = {};
  List<int> _sortedTrackedKeys = [];
  double _trackedFps = 30.0;
  final ValueNotifier<int> _uiPosition = ValueNotifier(0);

  List<PhysicsPoint> _speedData = [];
  List<PhysicsPoint> _posXData = [];
  List<PhysicsPoint> _posYData = [];
  List<PhysicsPoint> _absZData = [];
  List<PhysicsPoint> _filteredPixelHeightData = [];

  OrtSession? encoder;
  OrtSession? decoder;
  cv.Mat? template;
  late Rect templateBox;

  final TransformationController _transformationController = TransformationController();
  final GlobalKey _videoKey = GlobalKey();

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();
    _player.open(Media(widget.videoFile.path), play: false);
    loadMobileSam();

    _player.stream.tracks.listen((_) {
      if (mounted && !_isInitialized) _initIntrinsics();
    });
    _player.stream.width.listen((_) {
      if (mounted) setState(() {});
    });
    _player.stream.height.listen((_) {
      if (mounted) setState(() {});
    });
    Future.delayed(const Duration(milliseconds: 1500)).then((_) {
      if (mounted && !_isInitialized) _initIntrinsics();
    });

    _player.stream.error.listen((error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Video Error: $error"), backgroundColor: Colors.red),
        );
        setState(() => _isInitialized = true);
      }
    });

    _player.stream.position.listen((position) {
      if (!_isProcessing && !_isSeeking && mounted) {
        _uiPosition.value = position.inMilliseconds;
      }
    });

    _h.addListener(calculate);
    _fMM.addListener(calculate);
  }

  @override
  void dispose() {
    _player.dispose();
    template?.dispose();
    encoder?.release();
    decoder?.release();
    _transformationController.dispose();
    _h.dispose();
    _fMM.dispose();
    _samInterval.dispose();
    super.dispose();
  }

  // ===========================================================================
  // NON-UI METHODS (AI TRACKING, MPV FRAME STEPPING & 3D PHYSICS)
  // ===========================================================================

  Future<void> loadMobileSam() async {
    OrtEnv.instance.init();

    final encoderBytes = await rootBundle.load('assets/mobile_sam.encoder.onnx');
    final decoderBytes = await rootBundle.load('assets/mobile_sam.decoder.onnx');

    encoder = OrtSession.fromBuffer(encoderBytes.buffer.asUint8List(), OrtSessionOptions());
    decoder = OrtSession.fromBuffer(decoderBytes.buffer.asUint8List(), OrtSessionOptions());
  }

  List<Offset> getTrace(Uint8List bytes, Rect box, int width, int height) {
    cv.Mat frame = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, bytes);
    box = Rect.fromLTRB(
      (box.center.dx - box.width * 0.6).clamp(0.0, 1.0),
      (box.center.dy - box.height * 0.6).clamp(0.0, 1.0),
      (box.center.dx + box.width * 0.6).clamp(0.0, 1.0),
      (box.center.dy + box.height * 0.6).clamp(0.0, 1.0),
    );

    //10x square context area around bounding box
    int cropSize = (max(box.width * width, box.height * height) * 10.0).round().clamp(2, min(width, height));

    int cropX = (box.center.dx * width - cropSize / 2.0).round().clamp(0, width - cropSize);
    int cropY = (box.center.dy * height - cropSize / 2.0).round().clamp(0, height - cropSize);

    //resize to 1024x1024 and extract pixel rgb values
    cv.VecMat channels = cv.split(cv.resize(frame.region(cv.Rect(cropX, cropY, cropSize, cropSize)), (1024, 1024)));
    cv.Mat r = channels[0].convertTo(cv.MatType.CV_32FC1, alpha: 0.01712475, beta: -2.11790393);
    cv.Mat g = channels[1].convertTo(cv.MatType.CV_32FC1, alpha: 0.017507, beta: -2.03571429);
    cv.Mat b = channels[2].convertTo(cv.MatType.CV_32FC1, alpha: 0.01742919, beta: -1.80444444);

    //setup input to encoder
    Float32List input = Float32List(1 * 3 * 1024 * 1024);
    input.setAll(0, r.data.buffer.asFloat32List());
    input.setAll(1048576, g.data.buffer.asFloat32List());
    input.setAll(2097152, b.data.buffer.asFloat32List());

    for (cv.Mat c in channels) {
      c.dispose();
    }
    r.dispose();
    g.dispose();
    b.dispose();

    //run context window through encoder
    OrtValueTensor inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, 1024, 1024]);
    OrtValueTensor encoderOutputTensor = encoder!.run(OrtRunOptions(), {
      encoder!.inputNames[0]: inputTensor
    })[0] as OrtValueTensor;

    //search region for decoder to find trace
    Rect cropBox = Rect.fromLTRB(
      (box.left * width - cropX) / cropSize * 1024.0,
      (box.top * height - cropY) / cropSize * 1024.0,
      (box.right * width - cropX) / cropSize * 1024.0,
      (box.bottom * height - cropY) / cropSize * 1024.0,
    );

    //run decoder
    // 0: encoder output
    // 1: coordinates of points
    // 2: labels of inputted points (2.0 = top left, 3.0 = bottom right)
    // 3: previous mask (optional)
    // 4: if previous mask is used (1.0 if used, 0.0 if not)
    // 5: output mask size (directly scale back to original size)
    List<OrtValue?> decoderOutput = decoder!.run(OrtRunOptions(), {
      decoder!.inputNames[0]: encoderOutputTensor,
      decoder!.inputNames[1]: OrtValueTensor.createTensorWithDataList(Float32List.fromList([cropBox.left, cropBox.top, cropBox.right, cropBox.bottom]), [1, 2, 2]),
      decoder!.inputNames[2]: OrtValueTensor.createTensorWithDataList(Float32List.fromList([2.0, 3.0]), [1, 2]),
      decoder!.inputNames[3]: OrtValueTensor.createTensorWithDataList(Float32List(256 * 256), [1, 1, 256, 256]),
      decoder!.inputNames[4]: OrtValueTensor.createTensorWithDataList(Float32List(1), [1]),
      decoder!.inputNames[5]: OrtValueTensor.createTensorWithDataList(Float32List.fromList([cropSize.toDouble(), cropSize.toDouble()]), [2]),
    });
    List<List<double>> confidenceData = ((decoderOutput[0] as OrtValueTensor).value as List)[0][0];

    Uint8List binaryData = Uint8List(cropSize * cropSize);
    for (int y = 0; y < cropSize; y++) {
      for (int x = 0; x < cropSize; x++) {
        // greater than 0.0 = in object
        if (confidenceData[y][x] > 0.0) {
          //make object white
          binaryData[y * cropSize + x] = 255;
        }
      }
    }

    //trace outlines of white silhouettes
    cv.VecVecPoint contours = cv.findContours(cv.Mat.fromList(cropSize, cropSize, cv.MatType.CV_8UC1, binaryData), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE).$1;
    if (contours.isEmpty) return [];
    List<cv.VecPoint> sortedContours = List.generate(contours.length, (i) => contours[i])
      ..sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));

    //take greatest area outline
    cv.VecPoint bestContour = sortedContours.first;

    return [
      for (int j = 0; j < bestContour.length; j++)
        Offset((bestContour[j].x + cropX).toDouble(), (bestContour[j].y + cropY).toDouble()),
    ];
  }

  void initTracker(Uint8List bytes, Rect rec, int width, int height) {
    templateBox = rec;
    //rec contains decimals between 0 and 1, represents relative location in image
    int x = (rec.left * width).round().clamp(0, width - 2);
    int y = (rec.top * height).round().clamp(0, height - 2);
    int w = (rec.width * width).round().clamp(2, width - x);
    int h = (rec.height * height).round().clamp(2, height - y);
    //takes 2D template image from 1D bytes
    template = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, bytes).region(cv.Rect(x, y, w, h));
  }

  Rect trackObject(Uint8List bytes, int width, int height) {
    cv.Mat? nextFrame;
    nextFrame = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, bytes);

    //center of template box in the frame
    double cx = templateBox.center.dx * width;
    double cy = templateBox.center.dy * height;

    //dimensions of search area
    int searchW = (template!.cols * 5.0).ceil().clamp(0, width);
    int searchH = (template!.rows * 5.0).ceil().clamp(0, height);

    //top left of search area
    int searchX = (cx - searchW / 2.0).round().clamp(0, width - searchW);
    int searchY = (cy - searchH / 2.0).round().clamp(0, height - searchH);

    cv.Mat searchArea = nextFrame.region(cv.Rect(searchX, searchY, searchW, searchH));

    double bestConfidence = -1.0;
    cv.Point? bestLoc;
    double bestScale = 1.0;

    for (double scale = 0.8; scale <= 1.2; scale += 0.1) {
      int scaledW = (template!.cols * scale).round();
      int scaledH = (template!.rows * scale).round();

      if (scaledW >= searchArea.cols || scaledH >= searchArea.rows || scaledW < 2 || scaledH < 2) {
        continue;
      }

      cv.Mat scaledTemplate = cv.resize(template!, (scaledW, scaledH));
      //compare pixel brightness using cv.matchTemplate()
      final mx = cv.minMaxLoc(cv.matchTemplate(searchArea, scaledTemplate, cv.TM_CCOEFF_NORMED));
      if (mx.$2 > bestConfidence) {
        bestConfidence = mx.$2;
        bestLoc = mx.$4;
        bestScale = scale;
      }
      scaledTemplate.dispose();
    }

    //convert template back to decimal between 0 and 1
    templateBox = Rect.fromLTWH(
      ((searchX + bestLoc!.x) / width).clamp(0.0, 1.0),
      ((searchY + bestLoc.y) / height).clamp(0.0, 1.0),
      (bestScale * template!.cols / width).clamp(0.0, 1.0),
      (bestScale * template!.rows / height).clamp(0.0, 1.0),
    );

    nextFrame.dispose();
    searchArea.dispose();
    return templateBox;
  }

  Future<void> _initIntrinsics() async {
    if (_isInitialized) return;
    final double vidW = (_player.state.width ?? 0) > 0 ? _player.state.width!.toDouble() : 1280.0;
    final double vidH = (_player.state.height ?? 0) > 0 ? _player.state.height!.toDouble() : 720.0;

    _fMM.text = "26.0";
    if (widget.isRecordedInApp && Platform.isAndroid) {
      try {
        final Map<dynamic, dynamic>? hw = await const MethodChannel(
          'com.example.speed_tracker/camera_intrinsics',
        ).invokeMethod('getCameraIntrinsics', {
          'videoPath': widget.videoFile.path,
          'videoWidth': vidW.round(),
          'videoHeight': vidH.round(),
        });
        if (hw != null) {
          final double? hwF35 = (hw['focalLength35mm'] as num?)?.toDouble();
          if (hwF35 != null && hwF35 > 5.0) _fMM.text = hwF35.toStringAsFixed(1);
        }
      } catch (e) {
        debugPrint("Camera metadata extraction error: $e");
      }
    }

    if (mounted) setState(() => _isInitialized = true);
  }

  void _resetAll() {
    if (_player.state.playing) _player.pause();
    _player.seek(Duration.zero);
    setState(() {
      _isProcessing = false;
      _stopTrackingRequested = true;
      _startTrackMs = null;
      _endTrackMs = null;
      _rect = null;
      _boundaries = [];
      _trackedRects = {};
      _trackedBoundaries = {};
      _sortedTrackedKeys = [];
      _speedData = [];
      _posXData = [];
      _posYData = [];
      _absZData = [];
      _filteredPixelHeightData = [];
      _uiPosition.value = 0;
      _processingProgress = 0.0;
      _calculationResult = "Define range, draw box, then track.";
    });
  }

  Future<(Uint8List, int, int)?> _captureVideoFrame() async {
    final boundary = _videoKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
    if (boundary == null) return null;
    final double ratio = (1280.0 / math.max(10.0, boundary.size.longestSide)).clamp(2.0, 4.0);
    final ui.Image img = await boundary.toImage(pixelRatio: ratio);
    final ByteData? bd = await img.toByteData(format: ui.ImageByteFormat.rawRgba);
    final res = bd == null ? null : (bd.buffer.asUint8List(), img.width, img.height);
    img.dispose();
    return res;
  }

  Future<void> _traceFrame() async {
    if (_rect == null) return;
    setState(() => _isProcessing = true);
    try {
      final frame = await _captureVideoFrame();
      if (frame == null) return;
      final (bytes, width, height) = frame;
      final contour = getTrace(bytes, _rect!, width, height);
      setState(() {
        _boundaries = contour.map((p) => Offset(p.dx / width, p.dy / height)).toList();
        if (contour.isNotEmpty) _rect = _calculateTightRect(contour, width.toDouble(), height.toDouble());
      });
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  bool _areFramesIdentical(Uint8List a, Uint8List b) {
    if (a.length != b.length) return false;
    final int step = math.max(4, (a.length ~/ 400) & ~3);
    int diffSum = 0, count = 0;
    for (int i = 0; i < a.length; i += step) {
      diffSum += (a[i] - b[i]).abs();
      count++;
    }
    return count > 0 && (diffSum / count) < 1.0;
  }

  Future<void> _startLocalCVTracking() async {
    if (_isProcessing || _startTrackMs == null || _endTrackMs == null || _rect == null) return;

    setState(() {
      _isProcessing = true;
      _stopTrackingRequested = false;
      _trackedBoundaries.clear();
      _trackedRects.clear();
      _sortedTrackedKeys.clear();
      _processingProgress = 0.0;
      _boundaries = [];
    });

    try {
      if (_player.state.playing) await _player.pause();
      final int startMs = _startTrackMs!, endMs = _endTrackMs!;
      _trackedFps = _player.state.track.video.fps ?? 30.0;
      if (_trackedFps < 10.0) _trackedFps = 30.0;
      final double frameStepMs = 1000.0 / _trackedFps;

      if ((_player.state.position.inMilliseconds - startMs).abs() > 15) {
        await _player.seek(Duration(milliseconds: startMs));
        await Future.delayed(const Duration(milliseconds: 180));
      }
      await WidgetsBinding.instance.endOfFrame;

      Map<int, Rect> results = {};
      Rect currentBox = _rect!;
      int stepIndex = 0, framesSinceSam = 0;
      Uint8List? prevBytes;

      while (mounted && !_stopTrackingRequested) {
        final int currentMs = (startMs + stepIndex * frameStepMs).round();
        if (currentMs > endMs) break;

        (Uint8List, int, int)? frame;
        if (stepIndex == 0) {
          frame = await _captureVideoFrame();
        } else {
          try {
            await (_player.platform as dynamic).command(['frame-step']);
          } catch (_) {
            await _player.seek(Duration(milliseconds: currentMs));
          }
          for (int attempt = 0; attempt < 12; attempt++) {
            await Future.delayed(const Duration(milliseconds: 35));
            await WidgetsBinding.instance.endOfFrame;
            frame = await _captureVideoFrame();
            if (frame != null && (prevBytes == null || !_areFramesIdentical(frame.$1, prevBytes))) {
              break;
            }
            if (attempt == 6) {
              await _player.seek(Duration(milliseconds: currentMs));
            }
          }
        }

        if (frame == null) continue;
        stepIndex++;
        final (bytes, width, height) = frame;
        prevBytes = bytes;

        if (results.isNotEmpty) {
          currentBox = trackObject(bytes, width, height);
          framesSinceSam++;
        }

        final int samInterval = math.max(1, int.tryParse(_samInterval.text) ?? 1);
        if (results.isEmpty || framesSinceSam >= samInterval) {
          framesSinceSam = 0;
          final contour = getTrace(bytes, currentBox, width, height);
          if (contour.isNotEmpty) {
            _trackedBoundaries[currentMs] = contour.map((p) => Offset(p.dx / width, p.dy / height)).toList();
            currentBox = _calculateTightRect(contour, width.toDouble(), height.toDouble());
          }
          initTracker(bytes, currentBox, width, height);
        }

        results[currentMs] = currentBox;
        _uiPosition.value = currentMs;
        setState(() {
          _rect = currentBox;
          _processingProgress = ((currentMs - startMs) / math.max(1, endMs - startMs)).clamp(0.0, 1.0);
        });
      }

      setState(() {
        _rect = null;
        _trackedRects = results;
        _sortedTrackedKeys = results.keys.toList()..sort();
      });
      calculate();

      await _player.seek(Duration(milliseconds: startMs));
      _uiPosition.value = startMs;
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Rect _calculateTightRect(List<Offset> poly, double width, double height, {double scale = 1.0}) {
    if (poly.isEmpty) return _rect ?? Rect.zero;
    double minX = poly[0].dx, minY = poly[0].dy, maxX = poly[0].dx, maxY = poly[0].dy;
    for (final p in poly) {
      minX = math.min(minX, p.dx);
      minY = math.min(minY, p.dy);
      maxX = math.max(maxX, p.dx);
      maxY = math.max(maxY, p.dy);
    }
    return Rect.fromCenter(
      center: Offset((minX + maxX) / 2.0 / width, (minY + maxY) / 2.0 / height),
      width: (maxX - minX) * scale / width,
      height: (maxY - minY) * scale / height,
    );
  }

  void calculate() {
    double? objectHeight = double.tryParse(_h.text);
    double? fMM = double.tryParse(_fMM.text);
    int n = _sortedTrackedKeys.length;
    if (objectHeight == null || fMM == null || n < 3) return;

    double vidW = (_player.state.width ?? 0) > 0 ? _player.state.width!.toDouble() : 1280.0;
    double vidH = (_player.state.height ?? 0) > 0 ? _player.state.height!.toDouble() : 720.0;

    double cx = vidW / 2.0;
    double cy = vidH / 2.0;

    //pixel focal length
    double fPx = (fMM / 36.0) * math.max(vidW, vidH);

    double dt = 1.0 / (_trackedFps > 0 ? _trackedFps : 30.0);
    double t0 = _sortedTrackedKeys.first / 1000.0;

    //object locations
    List<Offset> centers = [
      for (int ms in _sortedTrackedKeys) Offset(_trackedRects[ms]!.center.dx * vidW, _trackedRects[ms]!.center.dy * vidH)
    ];

    List<double> times = List.generate(n, (i) => t0 + i * dt);
    List<double> filteredPixH = filterPixelHeight([for (final ms in _sortedTrackedKeys) _trackedRects[ms]!.height * vidH]);

    //calculate 3d position
    List<double> x = [], y = [], z = [];
    for (int i = 0; i < n; i++) {
      double metersPerPx = objectHeight / filteredPixH[i];
      x.add((centers[i].dx - cx) * metersPerPx);
      y.add((cy - centers[i].dy) * metersPerPx);
      z.add(fPx * metersPerPx);
    }

    //speed calculation using pythagorean theorem
    List<double> speeds = [];
    for (int i = 1; i < n; i++) {
      double dx = x[i] - x[i - 1];
      double dy = y[i] - y[i - 1];
      double dz = z[i] - z[i - 1];
      speeds.add(math.sqrt(dx * dx + dy * dy + dz * dz) / dt);
    }

    _updateGraphs(times, filteredPixH, x, y, z, speeds);
  }

  List<double> filterPixelHeight(List<double> rawPixH) {
    int n = rawPixH.length;

    //median filter to remove spikes
    List<double> medPixH = [];
    medPixH.add(rawPixH[0]);
    for(int i = 1; i < n - 1; i++){
      List<double> window = [rawPixH[i - 1], rawPixH[i], rawPixH[i + 1]]..sort();
      medPixH.add(window[1]);
    }
    medPixH.add(rawPixH[n - 1]);

    //for inverse quadratic, flip first and take quadratic regression
    List<double> invPixH = [for (double h in medPixH) 1.0 / h];

    //normalize times
    List<double> t = [for (int i = 0; i < n; i++) i / (n - 1)];

    double sumT0 = 0.0, sumT1 = 0.0, sumT2 = 0.0, sumT3 = 0.0, sumT4 = 0.0, sumHT0 = 0.0, sumHT1 = 0.0, sumHT2 = 0.0;
    //initialize known values
    for(int i = 0; i < n; i++){
      sumT0++;
      sumT1 += t[i];
      sumT2 += t[i] * t[i];
      sumT3 += t[i] * t[i] * t[i];
      sumT4 += t[i] * t[i] * t[i] * t[i];
      sumHT0 += invPixH[i];
      sumHT1 += invPixH[i] * t[i];
      sumHT2 += invPixH[i] * t[i] * t[i];
    }

    List<List<double>> mat = [
      [sumT0, sumT1, sumT2],
      [sumT1, sumT2, sumT3],
      [sumT2, sumT3, sumT4],
    ];

    List<List<double>> A = [
      [sumHT0, sumT1, sumT2],
      [sumHT1, sumT2, sumT3],
      [sumHT2, sumT3, sumT4],
    ];

    List<List<double>> B = [
      [sumT0, sumHT0, sumT2],
      [sumT1, sumHT1, sumT3],
      [sumT2, sumHT2, sumT4],
    ];

    List<List<double>> C = [
      [sumT0, sumT1, sumHT0],
      [sumT1, sumT2, sumHT1],
      [sumT2, sumT3, sumHT2],
    ];

    //3D determinant
    double det3D(List<List<double>> m) {
      return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
          - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
          + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    double a = det3D(A) / det3D(mat);
    double b = det3D(B) / det3D(mat);
    double c = det3D(C) / det3D(mat);

    return [for (int i = 0; i < n; i++) 1.0 / (a + b * t[i] + c * t[i] * t[i])];
  }

  int? _getClosestTrackedKey(int currentMs) {
    if (_sortedTrackedKeys.isEmpty) return null;
    int? bestKey;
    int bestDiff = 999999;
    for (int key in _sortedTrackedKeys) {
      final int diff = (key - currentMs).abs();
      if (diff < bestDiff) {
        bestDiff = diff;
        bestKey = key;
      }
    }
    return bestKey;
  }

  Rect? _getCurrentFrameRect(int currentMs) {
    if (_trackedRects.isEmpty || _sortedTrackedKeys.isEmpty) return null;
    if (currentMs <= _sortedTrackedKeys.first) return _trackedRects[_sortedTrackedKeys.first];
    if (currentMs >= _sortedTrackedKeys.last) return _trackedRects[_sortedTrackedKeys.last];

    int? tB, tA;
    for (int ms in _sortedTrackedKeys) {
      if (ms <= currentMs) {
        tB = ms;
      } else {
        tA = ms;
        break;
      }
    }
    if (tB == null) return _trackedRects[_sortedTrackedKeys.first];
    if (tA == null || tA == tB) return _trackedRects[tB];
    final double r = (currentMs - tB) / (tA - tB);
    final rB = _trackedRects[tB]!, rA = _trackedRects[tA]!;
    return Rect.fromLTRB(
      ui.lerpDouble(rB.left, rA.left, r)!,
      ui.lerpDouble(rB.top, rA.top, r)!,
      ui.lerpDouble(rB.right, rA.right, r)!,
      ui.lerpDouble(rB.bottom, rA.bottom, r)!,
    );
  }

  // ===========================================================================
  // UI METHODS (GRAPH DATA UPDATES, FORMATTING & WIDGET BUILDERS)
  // ===========================================================================

  void _updateGraphs(
    List<double> times,
    List<double> filteredPixH,
    List<double> x,
    List<double> y,
    List<double> z,
    List<double> speeds,
  ) {
    final int n = z.length;
    final List<PhysicsPoint> speedData = List.generate(
      speeds.length,
      (i) => PhysicsPoint(0.5 * (times[i] + times[i + 1]), speeds[i]),
    );
    final List<PhysicsPoint> xData = List.generate(n, (i) => PhysicsPoint(times[i], x[i] - x.first));
    final List<PhysicsPoint> yData = List.generate(n, (i) => PhysicsPoint(times[i], y[i] - y.first));
    final List<PhysicsPoint> absZData = List.generate(n, (i) => PhysicsPoint(times[i], z[i]));
    final List<PhysicsPoint> filtPixHData = List.generate(n, (i) => PhysicsPoint(times[i], filteredPixH[i]));

    setState(() {
      _speedData = speedData;
      _posXData = xData;
      _posYData = yData;
      _absZData = absZData;
      _filteredPixelHeightData = filtPixHData;
      if (speeds.isNotEmpty && times.last > times.first) {
        final double avgSpeed = speeds.reduce((a, b) => a + b) / speeds.length;
        final double peakSpeed = speeds.reduce(math.max);
        _calculationResult =
            "Avg: ${(avgSpeed * 2.23694).toStringAsFixed(1)} mph (${(avgSpeed * 3.6).toStringAsFixed(1)} kph) | "
            "Peak: ${(peakSpeed * 2.23694).toStringAsFixed(1)} mph (${(peakSpeed * 3.6).toStringAsFixed(1)} kph)";
      } else {
        _calculationResult = "Track object to see speed";
      }
    });
  }

  String _formatDuration(Duration d) {
    String tD(int n) => n.toString().padLeft(2, "0");
    return "${tD(d.inMinutes.remainder(60))}:${tD(d.inSeconds.remainder(60))}.${d.inMilliseconds.remainder(1000).toString().padLeft(3, '0')}";
  }

  String _formatSpeedStats(List<PhysicsPoint> data) {
    if (data.isEmpty) return '';
    final double avgMps = data.map((e) => e.value).reduce((a, b) => a + b) / data.length;
    final double peakMps = data.map((e) => e.value).reduce(math.max);
    return "  —  Avg: ${(avgMps * 2.23694).toStringAsFixed(1)} mph (${(avgMps * 3.6).toStringAsFixed(1)} kph) | "
        "Peak: ${(peakMps * 2.23694).toStringAsFixed(1)} mph (${(peakMps * 3.6).toStringAsFixed(1)} kph)";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Speed Tracker'), actions: _buildAppBarActions()),
      backgroundColor: Colors.black,
      body: !_isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(child: _showGraphs ? _buildGraphPanel() : _buildFrame()),
                if (_isProcessing)
                  LinearProgressIndicator(value: _processingProgress, color: Colors.purpleAccent),
                if (!_showGraphs) _buildMeasurementToolbar(),
                _buildResultPanel(),
              ],
            ),
    );
  }

  List<Widget> _buildAppBarActions() {
    return [
      IconButton(
        icon: Icon(Icons.edit, color: _isSelectionMode ? Colors.purpleAccent : Colors.white54),
        tooltip: "Selection Mode",
        onPressed: () => setState(() => _isSelectionMode = true),
      ),
      IconButton(
        icon: Icon(Icons.zoom_in, color: !_isSelectionMode ? Colors.blueAccent : Colors.white54),
        tooltip: "Zoom Mode",
        onPressed: () => setState(() => _isSelectionMode = false),
      ),
      IconButton(
        icon: Icon(
          _showGraphs ? Icons.video_library : Icons.bar_chart,
          color: _showGraphs ? Colors.greenAccent : Colors.white,
        ),
        tooltip: _showGraphs ? "View Video" : "View Graphs",
        onPressed: () => setState(() => _showGraphs = !_showGraphs),
      ),
    ];
  }

  Widget _buildMeasurementToolbar() {
    final bool canTrack =
        _startTrackMs != null && _endTrackMs != null && (_rect != null || _trackedRects.isNotEmpty);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      color: Colors.black,
      child: Row(
        children: [
          Expanded(
            child: TextButton.icon(
              onPressed: () => setState(() => _startTrackMs = _uiPosition.value),
              icon: Icon(
                Icons.start,
                color: _startTrackMs != null ? Colors.greenAccent : Colors.white54,
                size: 14,
              ),
              label: Text(
                _startTrackMs == null ? "In" : _formatDuration(Duration(milliseconds: _startTrackMs!)),
                style: const TextStyle(fontSize: 10),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),
          const SizedBox(width: 4),
          ElevatedButton.icon(
            onPressed: _isProcessing
                ? () => setState(() => _stopTrackingRequested = true)
                : (canTrack ? _startLocalCVTracking : null),
            icon: Icon(_isProcessing ? Icons.stop : Icons.troubleshoot, size: 16),
            label: Text(
              _isProcessing ? "STOP" : "TRACK",
              style: const TextStyle(fontWeight: FontWeight.w900, letterSpacing: 1.0, fontSize: 11),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: _isProcessing ? Colors.orangeAccent : Colors.cyanAccent,
              foregroundColor: Colors.black,
              disabledBackgroundColor: Colors.cyanAccent.withValues(alpha: 0.15),
              disabledForegroundColor: Colors.white38,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            ),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: TextButton.icon(
              onPressed: () => setState(() => _endTrackMs = _uiPosition.value),
              icon: Icon(
                Icons.output,
                color: _endTrackMs != null ? Colors.redAccent : Colors.white54,
                size: 14,
              ),
              label: Text(
                _endTrackMs == null ? "Out" : _formatDuration(Duration(milliseconds: _endTrackMs!)),
                style: const TextStyle(fontSize: 10),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white54, size: 18),
            tooltip: "Reset All",
            onPressed: _resetAll,
          ),
        ],
      ),
    );
  }

  Widget _buildFrame() {
    return Column(
      children: [
        Expanded(
          child: Stack(
            children: [
              InteractiveViewer(
                transformationController: _transformationController,
                panEnabled: !_isSelectionMode,
                scaleEnabled: !_isSelectionMode,
                minScale: 1.0,
                maxScale: 40.0,
                clipBehavior: Clip.none,
                boundaryMargin: const EdgeInsets.all(double.infinity),
                child: Center(
                  child: AspectRatio(
                    aspectRatio: ((_player.state.width ?? 0) > 0 && (_player.state.height ?? 0) > 0)
                        ? _player.state.width! / _player.state.height!
                        : 16 / 9,
                    child: IgnorePointer(
                      ignoring: !_isSelectionMode,
                      child: RectangleSelector(
                        rect: _rect,
                        enabled: _isSelectionMode,
                        onChanged: (nR) => setState(() {
                          _rect = nR;
                          if (nR != null) {
                            _trackedRects.clear();
                            _trackedBoundaries.clear();
                            _sortedTrackedKeys.clear();
                          }
                        }),
                        child: Stack(
                          children: [
                            RepaintBoundary(
                              key: _videoKey,
                              child: Video(controller: _controller, controls: NoVideoControls),
                            ),
                            ValueListenableBuilder<int>(
                              valueListenable: _uiPosition,
                              builder: (context, ms, _) {
                                final aR = _getCurrentFrameRect(ms);
                                final cK = _getClosestTrackedKey(ms);
                                final fB = (cK != null) ? _trackedBoundaries[cK] : null;
                                return Stack(
                                  children: [
                                    if (_sortedTrackedKeys.isNotEmpty)
                                      Positioned.fill(
                                        child: CustomPaint(
                                          size: Size.infinite,
                                          painter: TrajectoryPainter(_trackedRects, _sortedTrackedKeys, ms),
                                        ),
                                      ),
                                    if (aR != null)
                                      Positioned.fill(
                                        child: CustomPaint(
                                          size: Size.infinite,
                                          painter: _RectPainter(aR, color: Colors.cyanAccent, thickness: 3.0),
                                        ),
                                      ),
                                    if (fB != null)
                                      Positioned.fill(
                                        child: CustomPaint(
                                          size: Size.infinite,
                                          painter: BoundaryPainter(fB),
                                        ),
                                      ),
                                  ],
                                );
                              },
                            ),
                            if (_boundaries.isNotEmpty)
                              CustomPaint(
                                painter: BoundaryPainter(_boundaries),
                                size: Size.infinite,
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              Positioned(
                top: 8,
                left: 8,
                child: CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    icon: _isProcessing
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.psychology, color: Colors.white),
                    tooltip: "Preview SAM Trace",
                    onPressed: _isProcessing ? null : _traceFrame,
                  ),
                ),
              ),
            ],
          ),
        ),
        _buildScrubber(),
        _buildDimensionInputs(),
      ],
    );
  }

  Widget _buildScrubber() {
    return StreamBuilder<Duration>(
      stream: _player.stream.duration,
      builder: (context, snapshot) {
        final Duration duration = snapshot.data ?? _player.state.duration;
        final double max = math.max(100.0, duration.inMilliseconds.toDouble());

        return Container(
          color: Colors.grey.shade900,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          child: Row(
            children: [
              IconButton(
                icon: Icon(_player.state.playing ? Icons.pause : Icons.play_arrow),
                onPressed: () => setState(() => _player.state.playing ? _player.pause() : _player.play()),
              ),
              Expanded(
                child: ValueListenableBuilder<int>(
                  valueListenable: _uiPosition,
                  builder: (context, ms, _) {
                    final double sliderMax = math.max(max, ms.toDouble() + 1000.0);
                    final bool isLoaded = max > 100.0;

                    return Slider(
                      value: ms.toDouble().clamp(0.0, sliderMax),
                      min: 0.0,
                      max: sliderMax,
                      activeColor: Colors.purpleAccent,
                      onChangeStart: (_) {
                        _isSeeking = true;
                        if (_player.state.playing) _player.pause();
                      },
                      onChanged: isLoaded
                          ? (nV) {
                              final int tMs = nV.toInt();
                              _uiPosition.value = tMs;
                              if (!_seekInFlight) {
                                _seekInFlight = true;
                                _player.seek(Duration(milliseconds: tMs)).whenComplete(() {
                                  _seekInFlight = false;
                                });
                              }
                            }
                          : null,
                      onChangeEnd: (nV) async {
                        final int tMs = nV.toInt();
                        _uiPosition.value = tMs;
                        await _player.seek(Duration(milliseconds: tMs));
                        _seekInFlight = false;
                        _isSeeking = false;
                      },
                    );
                  },
                ),
              ),
              ValueListenableBuilder<int>(
                valueListenable: _uiPosition,
                builder: (context, ms, _) => Text(
                  _formatDuration(Duration(milliseconds: ms)),
                  style: const TextStyle(color: Colors.white70, fontSize: 10, fontFamily: 'monospace'),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildDimensionInputs() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      color: Colors.black,
      child: Row(
        children: [
          const Text("Height (m)", style: TextStyle(fontSize: 10, color: Colors.white70)),
          const SizedBox(width: 6),
          Expanded(
            child: TextField(
              controller: _h,
              decoration: const InputDecoration(hintText: "1.0", isDense: true),
              style: const TextStyle(fontSize: 11),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(width: 12),
          const Text("Focal (mm)", style: TextStyle(fontSize: 10, color: Colors.white70)),
          const SizedBox(width: 6),
          Expanded(
            child: TextField(
              controller: _fMM,
              decoration: const InputDecoration(hintText: "26.0", isDense: true),
              style: const TextStyle(fontSize: 11),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(width: 12),
          const Text("SAM (f)", style: TextStyle(fontSize: 10, color: Colors.white70)),
          const SizedBox(width: 6),
          Expanded(
            child: TextField(
              controller: _samInterval,
              enabled: !_isProcessing,
              decoration: const InputDecoration(hintText: "1", isDense: true),
              style: const TextStyle(fontSize: 11),
              keyboardType: TextInputType.number,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      color: Colors.grey.shade900,
      child: Text(
        _calculationResult,
        textAlign: TextAlign.center,
        style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold, fontSize: 12),
      ),
    );
  }

  Widget _buildGraphPanel() {
    if (_speedData.isEmpty) {
      return const Center(
        child: Text("No physics data yet.", style: TextStyle(color: Colors.white54)),
      );
    }
    return Container(
      color: Colors.black,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildGraphGroup("MOTION DYNAMICS", [
              _buildPlot("Speed vs Time (m/s)${_formatSpeedStats(_speedData)}", _speedData, Colors.yellowAccent),
            ]),
            const SizedBox(height: 24),
            _buildGraphGroup("3D SPATIAL COORDINATES & PIXEL HEIGHT VS TIME", [
              _buildPlot(
                "X(t) - Lateral (${_posXData.last.value.toStringAsFixed(2)} m)",
                _posXData,
                Colors.redAccent,
              ),
              _buildPlot(
                "Y(t) - Vertical (${_posYData.last.value.toStringAsFixed(2)} m)",
                _posYData,
                Colors.greenAccent,
              ),
              _buildPlot(
                "Z(t) - Depth (${_absZData.last.value.toStringAsFixed(2)} m)",
                _absZData,
                Colors.cyanAccent,
              ),
              _buildPlot(
                "Filtered Pixel Height (${_filteredPixelHeightData.last.value.toStringAsFixed(1)} px)",
                _filteredPixelHeightData,
                Colors.lightBlueAccent,
              ),
            ]),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildGraphGroup(String title, List<Widget> graphs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white38, letterSpacing: 1.1)),
        const SizedBox(height: 12),
        ...graphs.map((w) => Padding(padding: const EdgeInsets.only(bottom: 12), child: SizedBox(height: 100, child: w))),
      ],
    );
  }

  Widget _buildPlot(String title, List<PhysicsPoint> data, Color color) {
    return Column(
      children: [
        Text(title, style: const TextStyle(fontSize: 9, color: Colors.white70)),
        Expanded(
          child: Container(
            margin: const EdgeInsets.only(top: 4),
            decoration: BoxDecoration(border: Border.all(color: Colors.white10), color: Colors.black26),
            child: CustomPaint(size: Size.infinite, painter: GraphPainter(data, color)),
          ),
        ),
      ],
    );
  }
}

// =============================================================================
// DATA MODELS & CUSTOM PAINTERS / WIDGETS
// =============================================================================

class PhysicsPoint {
  final double time, value;
  PhysicsPoint(this.time, this.value);
}

class GraphPainter extends CustomPainter {
  final List<PhysicsPoint> data;
  final Color color;
  GraphPainter(this.data, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;
    double minV = data.map((e) => e.value).reduce(math.min);
    double maxV = data.map((e) => e.value).reduce(math.max);
    double minT = data.first.time;
    double maxT = data.last.time;
    if ((maxV - minV).abs() < 1e-4) maxV += 1.0;
    if ((maxT - minT).abs() < 1e-4) maxT += 1.0;

    final gridPaint = Paint()..color = Colors.white10..strokeWidth = 1.0;
    final double zeroY = (size.height - (0.0 - minV) / (maxV - minV) * size.height).clamp(0.0, size.height);
    canvas.drawLine(Offset(0, zeroY), Offset(size.width, zeroY), gridPaint);
    canvas.drawLine(Offset(0, size.height / 2), Offset(size.width, size.height / 2), gridPaint);

    final stemPaint = Paint()..color = color.withValues(alpha: 0.25)..strokeWidth = 1.0;
    final linePaint = Paint()..color = color.withValues(alpha: 0.5)..strokeWidth = 1.0..style = PaintingStyle.stroke;
    final dotPaint = Paint()..color = color..style = PaintingStyle.fill;

    final Path path = Path();
    for (int i = 0; i < data.length; i++) {
      final x = (data[i].time - minT) / (maxT - minT) * size.width;
      final y = size.height - (data[i].value - minV) / (maxV - minV) * size.height;

      canvas.drawLine(Offset(x, zeroY), Offset(x, y), stemPaint);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
      canvas.drawCircle(Offset(x, y), 3.0, dotPaint);
    }
    canvas.drawPath(path, linePaint);
  }

  @override
  bool shouldRepaint(covariant GraphPainter old) => true;
}

class RectangleSelector extends StatefulWidget {
  final Rect? rect;
  final Function(Rect?) onChanged;
  final Widget child;
  final bool enabled;

  const RectangleSelector({
    super.key,
    this.rect,
    required this.onChanged,
    required this.child,
    this.enabled = true,
  });

  @override
  State<RectangleSelector> createState() => _RectangleSelectorState();
}

class _RectangleSelectorState extends State<RectangleSelector> {
  Offset? _start;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final size = Size(constraints.maxWidth, constraints.maxHeight);
      return GestureDetector(
        onPanStart: widget.enabled ? (d) => setState(() => _start = d.localPosition) : null,
        onPanUpdate: widget.enabled
            ? (d) {
                if (_start == null) return;
                final rect = Rect.fromPoints(_start!, d.localPosition);
                widget.onChanged(Rect.fromLTRB(
                  (rect.left / size.width).clamp(0.0, 1.0),
                  (rect.top / size.height).clamp(0.0, 1.0),
                  (rect.right / size.width).clamp(0.0, 1.0),
                  (rect.bottom / size.height).clamp(0.0, 1.0),
                ));
              }
            : null,
        onPanEnd: (_) => setState(() => _start = null),
        onTapDown: widget.enabled
            ? (d) {
                if (widget.rect != null &&
                    !widget.rect!.contains(
                        Offset(d.localPosition.dx / size.width, d.localPosition.dy / size.height))) {
                  widget.onChanged(null);
                }
              }
            : null,
        child: Stack(
          fit: StackFit.expand,
          children: [
            widget.child,
            if (widget.rect != null) CustomPaint(painter: _RectPainter(widget.rect!)),
          ],
        ),
      );
    });
  }
}

class _RectPainter extends CustomPainter {
  final Rect rect;
  final Color color;
  final double thickness;
  _RectPainter(this.rect, {this.color = Colors.yellow, this.thickness = 2.0});

  @override
  void paint(Canvas canvas, Size size) {
    final r = Rect.fromLTRB(
      rect.left * size.width,
      rect.top * size.height,
      rect.right * size.width,
      rect.bottom * size.height,
    );
    canvas.drawRect(r, Paint()..color = color.withValues(alpha: 0.15));
    canvas.drawRect(r, Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = thickness);
  }

  @override
  bool shouldRepaint(covariant _RectPainter old) => true;
}

class BoundaryPainter extends CustomPainter {
  final List<Offset> points;
  BoundaryPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 2) return;
    final stroke = Paint()
      ..color = Colors.purpleAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.3;
    final fill = Paint()
      ..color = Colors.purpleAccent.withValues(alpha: 0.1)
      ..style = PaintingStyle.fill;
    final path = Path()..moveTo(points.first.dx * size.width, points.first.dy * size.height);
    for (int j = 1; j < points.length; j++) {
      path.lineTo(points[j].dx * size.width, points[j].dy * size.height);
    }
    path.close();
    canvas.drawPath(path, fill);
    canvas.drawPath(path, stroke);
  }

  @override
  bool shouldRepaint(covariant BoundaryPainter old) => true;
}

class TrajectoryPainter extends CustomPainter {
  final Map<int, Rect> trackedRects;
  final List<int> sortedKeys;
  final int currentMs;
  TrajectoryPainter(this.trackedRects, this.sortedKeys, this.currentMs);

  @override
  void paint(Canvas canvas, Size size) {
    if (sortedKeys.length < 2) return;
    final hPaint = Paint()
      ..color = Colors.cyanAccent.withValues(alpha: 0.6)
      ..strokeWidth = 3.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final path = Path();
    final firstR = trackedRects[sortedKeys[0]];
    if (firstR != null) path.moveTo(firstR.center.dx * size.width, firstR.center.dy * size.height);

    for (int i = 1; i < sortedKeys.length; i++) {
      final key = sortedKeys[i];
      if (key > currentMs) break;
      final r = trackedRects[key];
      if (r != null) path.lineTo(r.center.dx * size.width, r.center.dy * size.height);
    }
    canvas.drawPath(path, hPaint);

    final dPaint = Paint()
      ..color = Colors.cyanAccent.withValues(alpha: 0.8)
      ..style = PaintingStyle.fill;

    for (final key in sortedKeys) {
      if (key > currentMs) break;
      final r = trackedRects[key];
      if (r != null) canvas.drawCircle(Offset(r.center.dx * size.width, r.center.dy * size.height), 2.0, dPaint);
    }
  }

  @override
  bool shouldRepaint(covariant TrajectoryPainter old) => true;
}
