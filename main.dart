import 'dart:ui' as ui;
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/rendering.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'hq_sam_tracer.dart';
import 'opti_flow_tracker.dart';

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
      title: 'Dual Frame Speed Tracker',
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
            builder: (context) => VideoPlayerScreen(videoFile: File(file.path)),
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
              final picker = ImagePicker();
              final video = await picker.pickVideo(source: ImageSource.gallery);
              if (video != null) {
                if (!mounted) return;
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => VideoPlayerScreen(videoFile: File(video.path)),
                  ),
                );
              }
            },
          )
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

class VideoPlayerScreen extends StatefulWidget {
  final File videoFile;
  const VideoPlayerScreen({super.key, required this.videoFile});
  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late final Player _player = Player();
  late final VideoController _controller = VideoController(_player);

  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isSelectionMode = true;

  Rect? _rect;
  List<List<Offset>> _boundaries = [];
  int? _selectedTraceIndex;

  final TextEditingController _l = TextEditingController(text: "1.0");
  final TextEditingController _h = TextEditingController(text: "1.0");
  final TextEditingController _fMM = TextEditingController(text: "26.0");
  final TextEditingController _cx = TextEditingController();
  final TextEditingController _cy = TextEditingController();
  String _calculationResult = "Define range, draw box, then track.";

  double _processingProgress = 0.0;
  int? _startTrackMs;
  int? _endTrackMs;
  Map<int, Rect> _trackedRects = {};
  Map<int, List<List<Offset>>> _trackedBoundaries = {};
  List<int> _sortedTrackedKeys = [];
  bool _stopTrackingRequested = false;
  bool _isSeeking = false;
  bool _showGraphs = false;
  double _avgTraceAspectRatio = 1.0;
  bool _isSyncingInputs = false;
  final ValueNotifier<int> _uiPosition = ValueNotifier(0);

  List<PhysicsPoint> _velocityData = [];
  List<PhysicsPoint> _accelerationData = [];
  List<PhysicsPoint> _posXData = [];
  List<PhysicsPoint> _posYData = [];
  List<PhysicsPoint> _posZData = [];
  List<Offset3D> _positionData = [];

  final HqSamTracer _tracer = HqSamTracer();
  final OptiFlowTracker _flowTracker = OptiFlowTracker();
  final TransformationController _transformationController = TransformationController();
  final GlobalKey _videoKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _player.open(Media(widget.videoFile.path), play: false);

    _initAIModels();

    _player.stream.tracks.listen((tracks) {
      if (mounted && !_isInitialized) _initIntrinsics();
    });

    Future.delayed(const Duration(milliseconds: 1500)).then((_) {
      if (mounted && !_isInitialized) _initIntrinsics();
    });

    _player.stream.error.listen((error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Video Loading Error: $error"), backgroundColor: Colors.red),
        );
        setState(() => _isInitialized = true);
      }
    });

    _player.stream.position.listen((position) {
      // ONLY update UI position if we are NOT currently scrubbing or processing
      if (!_isProcessing && !_isSeeking && mounted) {
        _uiPosition.value = position.inMilliseconds;
      }
    });

    _l.addListener(() {
      if (!_isSyncingInputs && _sortedTrackedKeys.isNotEmpty && _avgTraceAspectRatio > 0) {
        _isSyncingInputs = true;
        final val = double.tryParse(_l.text);
        if (val != null) {
          _h.text = (val / _avgTraceAspectRatio).toStringAsFixed(3);
        }
        _isSyncingInputs = false;
      }
      _calculatePhysicsGraphs();
    });

    _h.addListener(() {
      if (!_isSyncingInputs && _sortedTrackedKeys.isNotEmpty && _avgTraceAspectRatio > 0) {
        _isSyncingInputs = true;
        final val = double.tryParse(_h.text);
        if (val != null) {
          _l.text = (val * _avgTraceAspectRatio).toStringAsFixed(3);
        }
        _isSyncingInputs = false;
      }
      _calculatePhysicsGraphs();
    });

    _fMM.addListener(_calculatePhysicsGraphs);
    _cx.addListener(_calculatePhysicsGraphs);
    _cy.addListener(_calculatePhysicsGraphs);
  }

  Future<void> _initAIModels() async {
    try {
      await _tracer.loadMobileSam();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Model Load Error: $e"), backgroundColor: Colors.red),
        );
      }
    }
  }

  void _initIntrinsics() {
    if (_isInitialized) return;
    final double vidW = (_player.state.width ?? 0) > 0 ? _player.state.width!.toDouble() : 1280.0;
    final double vidH = (_player.state.height ?? 0) > 0 ? _player.state.height!.toDouble() : 720.0;
    _cx.text = (vidW / 2).toStringAsFixed(1);
    _cy.text = (vidH / 2).toStringAsFixed(1);
    _fMM.text = "26.0";
    setState(() => _isInitialized = true);
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
      _velocityData = [];
      _accelerationData = [];
      _posXData = [];
      _posYData = [];
      _posZData = [];
      _positionData = [];
      _selectedTraceIndex = null;
      _uiPosition.value = 0;
      _processingProgress = 0.0;
      _calculationResult = "Define range, draw box, then track.";
    });
  }

  @override
  void dispose() {
    _player.dispose();
    _flowTracker.dispose();
    _tracer.dispose();
    _transformationController.dispose();
    _l.dispose();
    _h.dispose();
    _fMM.dispose();
    _cx.dispose();
    _cy.dispose();
    super.dispose();
  }

  Future<void> _traceFrame() async {
    if (_rect == null) return;
    setState(() => _isProcessing = true);
    try {
      final boundary = _videoKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) return;
      final image = await boundary.toImage(pixelRatio: 2.0);
      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      final bytes = byteData!.buffer.asUint8List();
      final prep = _tracer.preprocessRaw(bytes, image.width, image.height);
      List<List<Offset>> contours = await _tracer.extractMobileBoundaries(bytes, prep, roi: _rect!);
      final normalized = contours
          .map((poly) => poly
          .map((p) => Offset(p.dx / prep.originalSize.width, p.dy / prep.originalSize.height))
          .toList())
          .toList();
      setState(() {
        _boundaries = normalized;
        _selectedTraceIndex = null;
        
        // Snap the manual selection box to the tight fit of the AI trace
        if (contours.isNotEmpty && contours[0].isNotEmpty) {
          _rect = _calculateTightRect(contours[0], image.width.toDouble(), image.height.toDouble(), scale: 1.0);
        }
      });
      prep.dispose();
      image.dispose();
    } catch (e) {
      debugPrint("Trace Error: $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _startLocalCVTracking() async {
    if (_isProcessing || _startTrackMs == null || _endTrackMs == null || _rect == null) return;

    setState(() {
      _isProcessing = true;
      _stopTrackingRequested = false;
      _trackedBoundaries.clear();
      _trackedRects.clear();
      _sortedTrackedKeys = [];
      _processingProgress = 0.0;
      _boundaries = []; // Clear manual trace before starting loop
    });

    try {
      if (_player.state.playing) await _player.pause();
      final int startMs = _startTrackMs!, endMs = _endTrackMs!;
      await _player.seek(Duration(milliseconds: startMs));
      await Future.delayed(const Duration(milliseconds: 300));

      Map<int, Rect> results = {};
      Rect currentSearchBox = _rect!;
      
      // Clear out the tracking map keys list completely to ensure old data keys 
      // do not corrupt the physics graph indexing.
      _sortedTrackedKeys.clear();

      for (int t = startMs; t <= endMs; t += 33) {
        if (!mounted || _stopTrackingRequested) break;

        // Yield briefly to Flutter engine to keep UI responsive
        setState(() => _processingProgress = (t - startMs) / (endMs - startMs));
        await Future.delayed(const Duration(milliseconds: 5));

        await _player.seek(Duration(milliseconds: t));
        await Future.delayed(const Duration(milliseconds: 60));

        final loopBoundary = _videoKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
        if (loopBoundary == null) continue;

        final actualMs = _player.state.position.inMilliseconds;
        final loopImage = await loopBoundary.toImage(pixelRatio: 1.0);
        final loopByteData = await loopImage.toByteData(format: ui.ImageByteFormat.rawRgba);

        if (loopByteData == null) {
          loopImage.dispose();
          continue;
        }

        final loopBytes = loopByteData.buffer.asUint8List();

        // 1. Get predicted location from Template Matching (fast localized tracking)
        // If it's the very first frame, we skip running updateTracker since we already have 
        // a manual search box or a pre-calculated trace from _traceFrame().
        if (t == startMs) {
          _flowTracker.initTracker(loopBytes, currentSearchBox, [], width: loopImage.width, height: loopImage.height);
        }

        // Get the real native player timestamp of this exact video frame to align indices perfectly
        // (Removed duplicate actualMs variable declaration)

        Rect predictedBox = (t == startMs)
            ? currentSearchBox
            : (_flowTracker.updateTracker(loopBytes, width: loopImage.width, height: loopImage.height) ?? currentSearchBox);

        // Apply a 30% outer padding expansion onto the predictedBox ROI strictly when sending the prompt 
        // down to MobileSAM. This provides the AI with contrast context without inflating templates or calculations.
        // FIXED: Since MobileSAM's bounding box coordinate mapper expects a clean bounding format from the ROI 
        // bounding box, padding it symmetrically around the center using Rect.fromCenter can accidentally shift 
        // the top-left boundary into negative or unstable spaces during a bounce if the predictedBox size varies.
        // We pad it safely using the standard inflation method to preserve local aspect anchors.
        Rect paddedRoiForSam = Rect.fromLTRB(
          predictedBox.left - predictedBox.width * 0.15,
          predictedBox.top - predictedBox.height * 0.15,
          predictedBox.right + predictedBox.width * 0.15,
          predictedBox.bottom + predictedBox.height * 0.15,
        );

        // 2. Run MobileSAM ON EVERY FRAME for maximum proportional sizing & contour accuracy
        final prep = _tracer.preprocessRaw(loopBytes, loopImage.width, loopImage.height);
        try {
          List<List<Offset>> contours = await _tracer.extractMobileBoundaries(loopBytes, prep, roi: paddedRoiForSam);

          if (contours.isNotEmpty && contours[0].isNotEmpty) {
            final normalized = contours.map((poly) => poly.map((p) =>
                Offset(p.dx / prep.originalSize.width, p.dy / prep.originalSize.height)
            ).toList()).toList();

            // Keep the tracked baseline tightRect at exactly 1.0 scale to keep the recorded results 
            // and the yellow UI box perfectly pinned to the purple contour line without drifting.
            Rect tightRect = _calculateTightRect(contours[0], loopImage.width.toDouble(), loopImage.height.toDouble(), scale: 1.0);


            // VELOCITY MOMENTUM GUARD: 
            // SAM's result (tightRect) shouldn't be radically far from the 
            // Template Matcher's prediction (predictedBox).
            double predictedMoveDist = (predictedBox.center - currentSearchBox.center).distance;
            double actualDeviation = (tightRect.center - predictedBox.center).distance;
            
            // Reverted back to the original generous bounce-friendly limit (1.5x predicted movement or min 8% screen)
            // since the tight constraint was causing false rejections right at the sharp velocity switch frame.
            double maxAllowedDeviation = math.max(0.08, predictedMoveDist * 1.5);
            
            if (actualDeviation > maxAllowedDeviation) {
               debugPrint("SAM Wander Detected: $actualDeviation > $maxAllowedDeviation. Falling back to Template Match.");
               Rect adjustedBox = Rect.fromLTWH(
                 predictedBox.left,
                 predictedBox.top,
                 currentSearchBox.width,
                 currentSearchBox.height,
               );
               results[actualMs] = adjustedBox;
               _uiPosition.value = actualMs;
               setState(() {
                 _rect = adjustedBox;
               });
               currentSearchBox = adjustedBox;
               _flowTracker.initTracker(loopBytes, adjustedBox, [], width: loopImage.width, height: loopImage.height);
               continue; 
            }

            // Re-anchor Template Matching with the newly scaled bounds for the next frame
            // We feed the exact uninflated tightRect into the tracker initialization so the template 
            // remains perfectly cropped around the ball texture.
            _flowTracker.initTracker(loopBytes, tightRect, normalized, width: loopImage.width, height: loopImage.height);

            // However, we inflate currentSearchBox by a small scale factor ONLY when passing it down 
            // as the ROI bounding box to MobileSAM on the next frame's decoder input.
            currentSearchBox = tightRect;
            results[actualMs] = tightRect;
            _trackedBoundaries[actualMs] = normalized;

            // Update progress notifier directly (no setState needed for the value)
            _uiPosition.value = actualMs;

            setState(() {
              _rect = tightRect;
            });
          } else {
            // Fallback gracefully to the predicted box if SAM misses a frame
            Rect adjustedBox = Rect.fromLTWH(
              predictedBox.left,
              predictedBox.top,
              currentSearchBox.width,
              currentSearchBox.height,
            );
            results[actualMs] = adjustedBox;
            _uiPosition.value = actualMs;
            setState(() {
              _rect = adjustedBox;
            });
          }
        } catch (e) {
          debugPrint("SAM Error: $e");
        } finally {
          prep.dispose();
          loopImage.dispose();
        }
      }

      // Physics Velocity Correction:
      // The video player hardware seek operation (`_player.seek(...)`) takes a few milliseconds to settle. 
      // During the first few iterations, `_player.state.position.inMilliseconds` often reports the EXACT same 
      // timestamp (e.g. 1000ms followed by 1000ms again), which maps duplicate coordinates onto different frame intervals. 
      // This artificially introduced a zero-displacement initial step that killed the initial velocity calculation.
      // We explicitly enforce that the sequential tracking indices MUST increment monotonically.
      final correctedResults = <int, Rect>{};
      final correctedBoundaries = <int, List<List<Offset>>>{};
      
      int lastAddedMs = -9999;
      final sortedKeys = results.keys.toList()..sort();
      
      for (int ms in sortedKeys) {
        if (ms >= lastAddedMs + 15) { // Ensure at least a half-frame separation gap
          correctedResults[ms] = results[ms]!;
          if (_trackedBoundaries.containsKey(ms)) {
            correctedBoundaries[ms] = _trackedBoundaries[ms]!;
          }
          lastAddedMs = ms;
        }
      }

      if (correctedResults.isEmpty) throw Exception("Object lost immediately. No unique frames tracked.");

      setState(() {
        _trackedRects = correctedResults;
        _trackedBoundaries = correctedBoundaries;
        _sortedTrackedKeys = correctedResults.keys.toList()..sort();
        _calculatePhysicsGraphs();
      });

      await _player.seek(Duration(milliseconds: startMs));
    } catch (e) {
      debugPrint("Tracking Pipeline Error: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Error: $e"),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
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
    double curW = maxX - minX, curH = maxY - minY;
    
    // Scale the tight boundaries directly without forcing the original box's aspect ratio constraint
    double finalW = curW * scale;
    double finalH = curH * scale;
    
    return Rect.fromCenter(
      center: Offset((minX + curW / 2) / width, (minY + curH / 2) / height),
      width: finalW / width,
      height: finalH / height,
    );
  }

  void _calculatePhysicsGraphs() {
    final double? pW = double.tryParse(_l.text);
    final double? pH = double.tryParse(_h.text);
    final double? fMM = double.tryParse(_fMM.text);
    final double? cx = double.tryParse(_cx.text);
    final double? cy = double.tryParse(_cy.text);

    if (pW == null || pH == null || fMM == null || cx == null || cy == null || _sortedTrackedKeys.length < 2) {
      return;
    }

    final double vidW = (_player.state.width ?? 0) > 0 ? _player.state.width!.toDouble() : 1280.0;
    final double vidH = (_player.state.height ?? 0) > 0 ? _player.state.height!.toDouble() : 720.0;
    final double fx = (fMM / 36.0) * vidW, fy = fx;

    double totalRatio = 0;
    int ratioCount = 0;
    for (var frameContours in _trackedBoundaries.values) {
      if (frameContours.isNotEmpty) {
        final poly = frameContours[0];
        if (poly.isEmpty) continue;
        double minX = poly[0].dx, minY = poly[0].dy, maxX = poly[0].dx, maxY = poly[0].dy;
        for (final p in poly) {
          minX = math.min(minX, p.dx);
          minY = math.min(minY, p.dy);
          maxX = math.max(maxX, p.dx);
          maxY = math.max(maxY, p.dy);
        }
        double wPixels = (maxX - minX) * vidW;
        double hPixels = (maxY - minY) * vidH;
        if (wPixels > 0 && hPixels > 0) {
          totalRatio += wPixels / hPixels;
          ratioCount++;
        }
      }
    }
    if (ratioCount > 0 && !_isSyncingInputs) _avgTraceAspectRatio = totalRatio / ratioCount;

    List<Offset3D> positions = [];
    List<double> times = [];
    for (int ms in _sortedTrackedKeys) {
      double pixW, pixH, cX, cY;
      if (_trackedBoundaries.containsKey(ms) && _trackedBoundaries[ms]!.isNotEmpty) {
        final poly = _trackedBoundaries[ms]![0];
        double minX = poly[0].dx, minY = poly[0].dy, maxX = poly[0].dx, maxY = poly[0].dy;
        for (final p in poly) {
          minX = math.min(minX, p.dx);
          minY = math.min(minY, p.dy);
          maxX = math.max(maxX, p.dx);
          maxY = math.max(maxY, p.dy);
        }
        pixW = (maxX - minX) * vidW;
        pixH = (maxY - minY) * vidH;
        cX = (minX + (maxX - minX) / 2) * vidW;
        cY = (minY + (maxY - minY) / 2) * vidH;
      } else {
        final r = _trackedRects[ms]!;
        pixW = r.width * vidW;
        pixH = r.height * vidH;
        cX = r.center.dx * vidW;
        cY = r.center.dy * vidH;
      }
      final double z = ((fx * pW) / pixW + (fy * pH) / pixH) / 2.0;
      
      // INVERSION CORRECTION: Standard digital image coordinates have (0,0) at the Top-Left corner.
      // This means as an object moves physically DOWN (decreasing height/altitude), its 'cY' screen pixel index 
      // increases. We subtract cY from the camera height midpoint (cy) to align with standard physical Cartesian 
      // coordinates where upwards movement represents positive Y delta.
      positions.add(Offset3D((cX - cx) * z / fx, (cy - cY) * z / fy, z));
      times.add(ms / 1000.0);
    }

    List<Offset3D> sPos = [];
    for (int i = 0; i < positions.length; i++) {
      int s = math.max(0, i - 2), e = math.min(positions.length - 1, i + 2);
      double sx = 0, sy = 0, sz = 0;
      for (int j = s; j <= e; j++) {
        sx += positions[j].x;
        sy += positions[j].y;
        sz += positions[j].z;
      }
      sPos.add(Offset3D(sx / (e - s + 1), sy / (e - s + 1), sz / (e - s + 1)));
    }

    List<PhysicsPoint> rawVData = [];
    for (int i = 1; i < sPos.length; i++) {
      final dv = sPos[i].distanceTo(sPos[i - 1]), dt = times[i] - times[i - 1];
      if (dt > 0) {
        double instantV = dv / dt;
        if (instantV < 112.0) rawVData.add(PhysicsPoint(times[i], instantV));
      }
    }

    // Velocity Smoothing: 5-point moving average to eliminate derivative noise (pixel jitter)
    List<PhysicsPoint> vData = [];
    for (int i = 0; i < rawVData.length; i++) {
      int s = math.max(0, i - 2), e = math.min(rawVData.length - 1, i + 2);
      double sumV = 0;
      for (int j = s; j <= e; j++) sumV += rawVData[j].value;
      vData.add(PhysicsPoint(rawVData[i].time, sumV / (e - s + 1)));
    }

    // Acceleration Smoothing: Derived from smoothed velocity, then smoothed again
    List<PhysicsPoint> rawAData = [];
    for (int i = 1; i < vData.length; i++) {
      final dv = vData[i].value - vData[i - 1].value, dt = vData[i].time - vData[i - 1].time;
      if (dt > 0) rawAData.add(PhysicsPoint(vData[i].time, dv / dt));
    }

    List<PhysicsPoint> aData = [];
    for (int i = 0; i < rawAData.length; i++) {
      int s = math.max(0, i - 2), e = math.min(rawAData.length - 1, i + 2);
      double sumA = 0;
      for (int j = s; j <= e; j++) sumA += rawAData[j].value;
      aData.add(PhysicsPoint(rawAData[i].time, sumA / (e - s + 1)));
    }

    List<PhysicsPoint> xData = [], yData = [], zData = [];
    for (int i = 0; i < positions.length; i++) {
      xData.add(PhysicsPoint(times[i], positions[i].x));
      yData.add(PhysicsPoint(times[i], positions[i].y));
      zData.add(PhysicsPoint(times[i], positions[i].z));
    }

    setState(() {
      _velocityData = vData;
      _accelerationData = aData;
      _posXData = xData;
      _posYData = yData;
      _posZData = zData;
      _positionData = sPos; // Use smoothed path for 3D visualization
      if (vData.isNotEmpty) {
        double totalD = 0;
        // Use smoothed positions for average to remove distance added by pixel jitter
        for (int i = 1; i < sPos.length; i++) totalD += sPos[i].distanceTo(sPos[i - 1]);
        final pAvgV = totalD / (times.last - times.first);
        final peakV = vData.map((e) => e.value).reduce(math.max);
        _calculationResult =
        "Avg: ${(pAvgV * 2.23694).toStringAsFixed(1)} mph | Peak: ${(peakV * 2.23694).toStringAsFixed(1)} mph";
      } else {
        _calculationResult = "Track object to see speed";
      }
    });
  }

  int? _getClosestTrackedKey(int currentMs) {
    if (_sortedTrackedKeys.isEmpty) return null;
    int? tB;
    for (int key in _sortedTrackedKeys) {
      if (key <= currentMs) {
        tB = key;
      } else {
        break;
      }
    }
    return tB ?? _sortedTrackedKeys.first;
  }

  Rect? _getCurrentFrameRect(int currentMs) {
    if (_trackedRects.isEmpty || _sortedTrackedKeys.isEmpty) return null;
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
    if (tA == null) return _trackedRects[tB];
    final double r = (currentMs - tB) / (tA - tB);
    final rB = _trackedRects[tB]!, rA = _trackedRects[tA]!;
    return Rect.fromLTRB(
      ui.lerpDouble(rB.left, rA.left, r)!,
      ui.lerpDouble(rB.top, rA.top, r)!,
      ui.lerpDouble(rB.right, rA.right, r)!,
      ui.lerpDouble(rB.bottom, rA.bottom, r)!,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Speed Tracker Tracer'), actions: _buildAppBarActions()),
      backgroundColor: Colors.black,
      body: !_isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Column(
        children: [
          Expanded(
            child: _showGraphs
                ? _buildGraphPanel()
                : _buildFrame(_controller, _videoKey, _rect, _boundaries, _l, _h),
          ),
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
      Container(
        margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        decoration: BoxDecoration(
          color: _isSelectionMode ? Colors.purpleAccent : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: IconButton(
          icon: const Icon(Icons.edit),
          tooltip: "Selection Mode",
          onPressed: () => setState(() => _isSelectionMode = true),
        ),
      ),
      Container(
        margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        decoration: BoxDecoration(
          color: !_isSelectionMode ? Colors.blueAccent : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: IconButton(
          icon: const Icon(Icons.zoom_in),
          tooltip: "Zoom Mode",
          onPressed: () => setState(() => _isSelectionMode = false),
        ),
      ),
      Container(
        margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        decoration: BoxDecoration(
          color: _showGraphs ? Colors.greenAccent : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: IconButton(
          icon: Icon(_showGraphs ? Icons.video_library : Icons.bar_chart),
          tooltip: _showGraphs ? "View Video" : "View Graphs",
          onPressed: () => setState(() => _showGraphs = !_showGraphs),
        ),
      ),
      const SizedBox(width: 8),
    ];
  }

  Widget _buildMeasurementToolbar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
      color: Colors.black,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          TextButton.icon(
            onPressed: () => setState(() => _startTrackMs = _player.state.position.inMilliseconds),
            icon: Icon(
              Icons.start,
              color: _startTrackMs != null ? Colors.greenAccent : Colors.white54,
              size: 16,
            ),
            label: Text(
              _startTrackMs == null
                  ? "Set Start"
                  : "Start: ${_formatDuration(Duration(milliseconds: _startTrackMs!))}",
              style: const TextStyle(fontSize: 10),
            ),
          ),
          TextButton.icon(
            onPressed: () => setState(() => _endTrackMs = _player.state.position.inMilliseconds),
            icon: Icon(
              Icons.output,
              color: _endTrackMs != null ? Colors.redAccent : Colors.white54,
              size: 16,
            ),
            label: Text(
              _endTrackMs == null
                  ? "Set End"
                  : "End: ${_formatDuration(Duration(milliseconds: _endTrackMs!))}",
              style: const TextStyle(fontSize: 10),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white54, size: 20),
            onPressed: _resetAll,
          ),
        ],
      ),
    );
  }

  Widget _buildFrame(
      VideoController c,
      GlobalKey key,
      Rect? r,
      List<List<Offset>> b,
      TextEditingController l,
      TextEditingController h,
      ) {
    return Column(
      children: [
        Expanded(
          child: Stack(children: [
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
                          _selectedTraceIndex = null;
                          _trackedRects.clear();
                        }
                      }),
                      onTapTrace: (idx) => setState(() =>
                      _selectedTraceIndex = (_selectedTraceIndex == idx) ? null : idx),
                      tracePoints: _boundaries,
                      child: Stack(children: [
                        RepaintBoundary(
                          key: _videoKey,
                          child: Video(controller: c, controls: NoVideoControls),
                        ),
                        ValueListenableBuilder(
                          valueListenable: _uiPosition,
                          builder: (context, ms, child) {
                            final aR = _getCurrentFrameRect(ms);
                            final cK = _getClosestTrackedKey(ms);
                            final fB = (cK != null) ? _trackedBoundaries[cK] : null;
                            return Stack(children: [
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
                                    painter: _RectPainter(aR, color: Colors.cyanAccent, thickness: 4.0),
                                  ),
                                ),
                              if (fB != null)
                                Positioned.fill(
                                  child: CustomPaint(
                                    size: Size.infinite,
                                    painter: BoundaryPainter(fB),
                                  ),
                                ),
                            ]);
                          },
                        ),
                        if (_boundaries.isNotEmpty)
                          CustomPaint(
                            painter: BoundaryPainter(_boundaries, selectedIndex: _selectedTraceIndex),
                            size: Size.infinite,
                          ),
                      ]),
                    ),
                  ),
                ),
              ),
            ),
            Positioned(
              top: 8,
              left: 8,
              child: Row(children: [
                CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    icon: _isProcessing
                        ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                        : const Icon(Icons.psychology, color: Colors.white),
                    tooltip: "Refine Trace",
                    onPressed: _isProcessing ? null : () => _traceFrame(),
                  ),
                ),
                const SizedBox(width: 8),
                CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    icon: _isProcessing
                        ? const Icon(Icons.pause, color: Colors.orangeAccent)
                        : const Icon(Icons.track_changes, color: Colors.cyanAccent),
                    tooltip: _isProcessing ? "Stop Tracking" : "Start CV Tracking",
                    onPressed: _isProcessing
                        ? () => setState(() => _stopTrackingRequested = true)
                        : () => _startLocalCVTracking(),
                  ),
                ),
                const SizedBox(width: 8),
                if (_boundaries.isNotEmpty || _trackedRects.isNotEmpty || _startTrackMs != null)
                  CircleAvatar(
                    backgroundColor: Colors.black54,
                    child: IconButton(
                      icon: const Icon(Icons.layers_clear, color: Colors.redAccent),
                      onPressed: _resetAll,
                    ),
                  ),
              ]),
            ),
          ]),
        ),
        _buildScrubber(),
        _buildDimensionInputs(_l, _h, "Reference Object Size"),
      ],
    );
  }

  Widget _buildGraphPanel() {
    if (_velocityData.isEmpty) {
      return const Center(
        child: Text("No physics data yet.", style: TextStyle(color: Colors.white54)),
      );
    }
    return Container(
      color: Colors.black,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      child: Column(children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          const Text(
            "PHYSICS DASHBOARD",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w900,
              color: Colors.cyanAccent,
              letterSpacing: 1.2,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, color: Colors.white54),
            onPressed: () => setState(() => _showGraphs = false),
          ),
        ]),
        Expanded(
          child: SingleChildScrollView(
            child: Column(children: [
              const SizedBox(height: 20),
              _build3DTrajectoryView(),
              const SizedBox(height: 30),
              _buildGraphGroup("3D POSITION VECTORS (Meters)", [
                _buildGraph("X - Horizontal", _posXData, Colors.redAccent),
                _buildGraph("Y - Vertical", _posYData, Colors.greenAccent),
                _buildGraph("Z - Depth", _posZData, Colors.blueAccent),
              ]),
              const SizedBox(height: 30),
              _buildGraphGroup("MOTION DYNAMICS", [
                _buildGraph("Velocity (m/s)", _velocityData, Colors.cyanAccent),
                _buildGraph("Acceleration (m/s²)", _accelerationData, Colors.orangeAccent),
              ]),
              const SizedBox(height: 30),
              _buildIntrinsicsEditor(),
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(children: [
                  const Text(
                    "AVERAGE SPEED",
                    style: TextStyle(fontSize: 12, color: Colors.white38, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    _calculationResult,
                    style: const TextStyle(
                      fontSize: 28,
                      color: Colors.greenAccent,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ]),
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: () => setState(() => _showGraphs = false),
                icon: const Icon(Icons.video_library),
                label: const Text("RETURN TO VIDEO"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.cyan.shade900,
                  minimumSize: const Size(double.infinity, 50),
                ),
              ),
              const SizedBox(height: 20),
            ]),
          ),
        ),
      ]),
    );
  }

  Widget _build3DTrajectoryView() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text(
        "3D SPATIAL RECONSTRUCTION",
        style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white38, letterSpacing: 1.1),
      ),
      const SizedBox(height: 12),
      const Text("Swipe to rotate view", style: TextStyle(fontSize: 9, color: Colors.white24)),
      const SizedBox(height: 8),
      ThreeDTrajectoryView(points: _positionData),
    ]);
  }

  Widget _buildIntrinsicsEditor() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text(
        "CAMERA INTRINSIC PARAMETERS (Testing)",
        style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white38, letterSpacing: 1.1),
      ),
      const SizedBox(height: 12),
      Row(children: [
        Expanded(child: _buildSmallInput("f (mm)", _fMM)),
        const SizedBox(width: 8),
        Expanded(child: _buildSmallInput("cx", _cx)),
        const SizedBox(width: 8),
        Expanded(child: _buildSmallInput("cy", _cy)),
      ]),
    ]);
  }

  Widget _buildSmallInput(String l, TextEditingController c) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(l, style: const TextStyle(fontSize: 9, color: Colors.white24)),
      TextField(
        controller: c,
        decoration: const InputDecoration(isDense: true, contentPadding: EdgeInsets.symmetric(vertical: 8)),
        style: const TextStyle(fontSize: 11, color: Colors.white70),
        keyboardType: TextInputType.number,
      )
    ]);
  }

  Widget _buildGraphGroup(String t, List<Widget> g) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(t, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white38, letterSpacing: 1.1)),
      const SizedBox(height: 12),
      ...g.map((w) => Padding(padding: const EdgeInsets.only(bottom: 12), child: SizedBox(height: 100, child: w))),
    ]);
  }

  Widget _buildScrubber() {
    return StreamBuilder<Duration>(
      stream: _player.stream.duration,
      builder: (context, snapshot) {
        // Fallback to player state if stream hasn't emitted yet
        final Duration duration = snapshot.data ?? _player.state.duration;
        double max = duration.inMilliseconds.toDouble();
        if (max <= 0) max = 100.0; // Temporary default
        
        return Container(
          color: Colors.grey.shade900,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          child: Row(children: [
            IconButton(
              icon: Icon(_player.state.playing ? Icons.pause : Icons.play_arrow),
              onPressed: () => setState(() => _player.state.playing ? _player.pause() : _player.play()),
            ),
            Expanded(
              child: ValueListenableBuilder(
                valueListenable: _uiPosition,
                builder: (context, ms, child) {
                  // If duration isn't loaded yet, but we have a position, push 'max' out
                  double sliderMax = max;
                  if (ms.toDouble() > max) {
                    sliderMax = ms.toDouble() + 1000.0;
                  }
                  
                  final bool isLoaded = max > 100.0; // Assume >100ms means real metadata loaded

                  return SliderTheme(
                    data: SliderTheme.of(context).copyWith(
                      trackHeight: 4,
                      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
                      activeTrackColor: isLoaded ? Colors.purpleAccent : Colors.grey,
                      inactiveTrackColor: Colors.white24,
                      thumbColor: isLoaded ? Colors.purpleAccent : Colors.grey,
                    ),
                    child: Slider(
                      value: ms.toDouble().clamp(0, sliderMax),
                      min: 0,
                      max: sliderMax,
                      divisions: (sliderMax > 0) ? (sliderMax / 10).round().clamp(1, 1000) : 1,
                      onChanged: isLoaded ? (nV) {
                        final tMs = nV.toInt();
                        _uiPosition.value = tMs;
                        if (!_isSeeking) {
                          _isSeeking = true;
                          _player.seek(Duration(milliseconds: tMs)).then((_) {
                            if (mounted) {
                              setState(() => _isSeeking = false);
                            }
                          });
                        }
                      } : null,
                    ),
                  );
                },
              ),
            ),
            ValueListenableBuilder(
              valueListenable: _uiPosition,
              builder: (context, ms, child) => Text(
                _formatDuration(Duration(milliseconds: ms)),
                style: const TextStyle(color: Colors.white70, fontSize: 10, fontFamily: 'monospace'),
              ),
            ),
          ]),
        );
      },
    );
  }

  String _formatDuration(Duration d) {
    String tD(int n) => n.toString().padLeft(2, "0");
    return "${tD(d.inMinutes.remainder(60))}:${tD(d.inSeconds.remainder(60))}.${d.inMilliseconds.remainder(1000).toString().padLeft(3, '0')}";
  }

  Widget _buildDimensionInputs(TextEditingController l, TextEditingController h, String lbl) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      color: Colors.black,
      child: Column(children: [
        Row(children: [
          Text(lbl, style: const TextStyle(fontSize: 10, color: Colors.white70)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: l,
              enabled: _sortedTrackedKeys.isNotEmpty,
              decoration: const InputDecoration(hintText: "Width (m)", isDense: true),
              style: TextStyle(fontSize: 11, color: _sortedTrackedKeys.isEmpty ? Colors.white24 : Colors.white),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: h,
              enabled: _sortedTrackedKeys.isNotEmpty,
              decoration: const InputDecoration(hintText: "Height (m)", isDense: true),
              style: TextStyle(fontSize: 11, color: _sortedTrackedKeys.isEmpty ? Colors.white24 : Colors.white),
              keyboardType: TextInputType.number,
            ),
          ),
        ]),
        const SizedBox(height: 4),
        Row(children: [
          const Text("Lens Focal Length (mm eq)", style: TextStyle(fontSize: 10, color: Colors.white70)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: _fMM,
              decoration: const InputDecoration(hintText: "e.g. 26.0", isDense: true),
              style: const TextStyle(fontSize: 11),
              keyboardType: TextInputType.number,
            ),
          ),
          const Spacer(),
        ]),
      ]),
    );
  }

  Widget _buildResultPanel() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.grey.shade900,
      child: Column(children: [
        Text(
          _calculationResult,
          style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold, fontSize: 12),
        ),
        if (_velocityData.isNotEmpty) ...[
          const SizedBox(height: 8),
          SizedBox(
            height: 80,
            child: Row(children: [
              Expanded(child: _buildGraph("Velocity (m/s)", _velocityData, Colors.cyanAccent)),
              const SizedBox(width: 8),
              Expanded(child: _buildGraph("Accel (m/s²)", _accelerationData, Colors.orangeAccent)),
            ]),
          )
        ],
      ]),
    );
  }

  Widget _buildGraph(String t, List<PhysicsPoint> d, Color c) {
    return Column(children: [
      Text(t, style: const TextStyle(fontSize: 9, color: Colors.white70)),
      Expanded(
        child: Container(
          margin: const EdgeInsets.only(top: 4),
          decoration: BoxDecoration(border: Border.all(color: Colors.white10), color: Colors.black26),
          child: CustomPaint(size: Size.infinite, painter: GraphPainter(d, c)),
        ),
      )
    ]);
  }
}

class PhysicsPoint {
  final double time, value;
  PhysicsPoint(this.time, this.value);
}

class Offset3D {
  final double x, y, z;
  Offset3D(this.x, this.y, this.z);
  double distanceTo(Offset3D o) =>
      math.sqrt(math.pow(x - o.x, 2) + math.pow(y - o.y, 2) + math.pow(z - o.z, 2));
}

class GraphPainter extends CustomPainter {
  final List<PhysicsPoint> data;
  final Color color;
  GraphPainter(this.data, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;
    double minV = data.map((e) => e.value).reduce(math.min);
    double maxV = data.map((e) => e.value).reduce(math.max);
    double minT = data.first.time;
    double maxT = data.last.time;
    if (maxV == minV) maxV += 1.0;
    if (maxT == minT) maxT += 1.0;

    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (int i = 0; i < data.length; i++) {
      final x = (data[i].time - minT) / (maxT - minT) * size.width;
      final y = size.height - (data[i].value - minV) / (maxV - minV) * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant GraphPainter old) => true;
}

class ThreeDTrajectoryView extends StatefulWidget {
  final List<Offset3D> points;
  const ThreeDTrajectoryView({super.key, required this.points});
  @override
  State<ThreeDTrajectoryView> createState() => _ThreeDTrajectoryViewState();
}

class _ThreeDTrajectoryViewState extends State<ThreeDTrajectoryView> {
  double _phi = 0.5, _theta = 0.5;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (d) => setState(() {
        _phi += d.delta.dx * 0.01;
        _theta += d.delta.dy * 0.01;
      }),
      child: Container(
        height: 250,
        width: double.infinity,
        decoration: BoxDecoration(
          color: Colors.black,
          border: Border.all(color: Colors.cyanAccent.withValues(alpha: 0.1)),
          borderRadius: BorderRadius.circular(16),
        ),
        child: ClipRect(
          child: CustomPaint(
            painter: ThreeDPainter(widget.points, _phi, _theta),
            size: Size.infinite,
          ),
        ),
      ),
    );
  }
}

class ThreeDPainter extends CustomPainter {
  final List<Offset3D> points;
  final double phi, theta;
  ThreeDPainter(this.points, this.phi, this.theta);

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) return;
    final center = Offset(size.width / 2, size.height / 2);
    const double scale = 50.0;

    Offset project(Offset3D p) {
      double x1 = p.x * math.cos(phi) - p.z * math.sin(phi);
      double z1 = p.x * math.sin(phi) + p.z * math.cos(phi);
      double y2 = p.y * math.cos(theta) - z1 * math.sin(theta);
      return Offset(center.dx + x1 * scale, center.dy + y2 * scale);
    }

    final axisPaint = Paint()..strokeWidth = 1.0;
    void drawAxis(Offset3D end, Color color) {
      canvas.drawLine(center, project(end), axisPaint..color = color.withValues(alpha: 0.5));
    }

    drawAxis(Offset3D(2, 0, 0), Colors.red);
    drawAxis(Offset3D(0, 2, 0), Colors.green);
    drawAxis(Offset3D(0, 0, 2), Colors.blue);

    final linePaint = Paint()
      ..color = Colors.cyanAccent
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final path = Path();
    for (int i = 0; i < points.length; i++) {
      final p = project(points[i]);
      if (i == 0) {
        path.moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }
    canvas.drawPath(path, linePaint);
    canvas.drawCircle(project(points.first), 4.0, Paint()..color = Colors.greenAccent);
    canvas.drawCircle(project(points.last), 4.0, Paint()..color = Colors.redAccent);
  }

  @override
  bool shouldRepaint(covariant ThreeDPainter old) => true;
}

class RectangleSelector extends StatefulWidget {
  final Rect? rect;
  final Function(Rect?) onChanged;
  final Function(int)? onTapTrace;
  final List<List<Offset>> tracePoints;
  final Widget child;
  final bool enabled;

  const RectangleSelector({
    super.key,
    this.rect,
    required this.onChanged,
    this.onTapTrace,
    required this.tracePoints,
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
          if (widget.tracePoints.isNotEmpty && widget.onTapTrace != null) {
            final tapPos = Offset(d.localPosition.dx / size.width, d.localPosition.dy / size.height);
            for (int i = 0; i < widget.tracePoints.length; i++) {
              final poly = widget.tracePoints[i];
              final path = Path();
              if (poly.isNotEmpty) {
                path.moveTo(poly[0].dx, poly[0].dy);
                for (var j = 1; j < poly.length; j++) path.lineTo(poly[j].dx, poly[j].dy);
                path.close();
                if (path.contains(tapPos)) {
                  widget.onTapTrace!(i);
                  return;
                }
              }
            }
          }
          if (widget.rect != null &&
              !widget.rect!.contains(
                  Offset(d.localPosition.dx / size.width, d.localPosition.dy / size.height))) {
            widget.onChanged(null);
          }
        }
            : null,
        child: Stack(fit: StackFit.expand, children: [
          widget.child,
          if (widget.rect != null) CustomPaint(painter: _RectPainter(widget.rect!))
        ]),
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
    if (color == Colors.cyanAccent) {
      canvas.drawRect(
        r,
        Paint()
          ..color = color.withValues(alpha: 0.3)
          ..style = PaintingStyle.stroke
          ..strokeWidth = thickness + 2
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _RectPainter old) => true;
}

class BoundaryPainter extends CustomPainter {
  final List<List<Offset>> points;
  final int? selectedIndex;
  BoundaryPainter(this.points, {this.selectedIndex});

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..color = Colors.purpleAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.3;

    for (int i = 0; i < points.length; i++) {
      final poly = points[i];
      if (poly.length < 2) continue;
      final fill = Paint()
        ..color = Colors.purpleAccent.withValues(alpha: 0.1)
        ..style = PaintingStyle.fill;
      final path = Path();
      path.moveTo(poly.first.dx * size.width, poly.first.dy * size.height);
      for (int j = 1; j < poly.length; j++) {
        path.lineTo(poly[j].dx * size.width, poly[j].dy * size.height);
      }
      path.close();
      canvas.drawPath(path, fill);
      canvas.drawPath(path, stroke);
    }
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
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2);

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