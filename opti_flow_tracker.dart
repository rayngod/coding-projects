import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class OptiFlowTracker {
  cv.Mat? _template;
  cv.Mat? _anchorTemplate; 
  Rect? _currentBox;
  double _vx = 0.0;
  double _vy = 0.0;
  int _consecutiveLost = 0; // NEW: Count lost frames for search expansion
  bool _isInitialized = false;

  /// Initialize using the initial bounding box / SAM trace safely
  void initTracker(Uint8List frameBytes, Rect userBox, List<List<Offset>> samTrace, {int? width, int? height}) {
    try {
      cv.Mat mat;
      if (width != null && height != null) {
        final raw = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, frameBytes);
        mat = cv.cvtColor(raw, cv.COLOR_RGBA2GRAY);
        raw.dispose();
      } else {
        mat = cv.imdecode(frameBytes, cv.IMREAD_GRAYSCALE);
      }

      if (mat.isEmpty) return;
      final int imgW = mat.cols;
      final int imgH = mat.rows;

      int x = (userBox.left * imgW).round().clamp(0, imgW - 2);
      int y = (userBox.top * imgH).round().clamp(0, imgH - 2);
      int w = (userBox.width * imgW).round().clamp(2, imgW - x);
      int h = (userBox.height * imgH).round().clamp(2, imgH - y);

      if (x + w > imgW) w = imgW - x;
      if (y + h > imgH) h = imgH - y;

      final roi = cv.Rect(x, y, w, h);
      _template?.dispose();
      _template = mat.region(roi);
      
      // MASTER ANCHOR PROTECTION:
      // Only capture the master anchor on the first frame initialization.
      // This preserves a pristine, high-contrast reference of the actual ball
      // and stops it from learning court lines/shadows over time.
      if (!_isInitialized || _anchorTemplate == null) {
        _anchorTemplate?.dispose();
        _anchorTemplate = _template!.clone();
      }

      // IMPORTANT: When re-anchoring from SAM every frame, we do NOT want to 
      // wipe out the momentum (_vx, _vy) already calculated by updateTracker.
      // We only set velocity to 0 if this is the very first initialization.
      if (!_isInitialized) {
        _vx = 0.0;
        _vy = 0.0;
      }

      _currentBox = userBox;
      _consecutiveLost = 0;
      _isInitialized = true;
      mat.dispose();
    } catch (e) {
      debugPrint("InitTracker Safe Error: $e");
      _isInitialized = false;
    }
  }

  /// Update position safely with strict boundary clamps
  Rect? updateTracker(Uint8List nextFrameBytes, {int? width, int? height}) {
    if (!_isInitialized || _template == null || _currentBox == null) return null;

    cv.Mat? nextMat;
    cv.Mat? searchRoi;

    try {
      if (width != null && height != null) {
        final raw = cv.Mat.fromList(height, width, cv.MatType.CV_8UC4, nextFrameBytes);
        nextMat = cv.cvtColor(raw, cv.COLOR_RGBA2GRAY);
        raw.dispose();
      } else {
        nextMat = cv.imdecode(nextFrameBytes, cv.IMREAD_GRAYSCALE);
      }

      if (nextMat.isEmpty) {
        nextMat.dispose();
        return null;
      }

      final imgW = nextMat.cols;
      final imgH = nextMat.rows;

      // 1. OMNI-DIRECTIONAL BOUNCE WINDOW
      // We search around the last known position. The window must be large enough
      // to cover the displacement (forward or bounce) plus a safety margin.
      double centerPxX = (_currentBox!.left + _currentBox!.width / 2.0) * imgW;
      double centerPxY = (_currentBox!.top + _currentBox!.height / 2.0) * imgH;
      
      double dx = (_vx * imgW).abs();
      double dy = (_vy * imgH).abs();
      double ballW = _currentBox!.width * imgW;
      double ballH = _currentBox!.height * imgH;

      // Tight, displacement-aware search window.
      // If lost, we expand to a large local search.
      double marginScale = (_consecutiveLost > 0) ? 10.0 : 4.5;
      int searchW = (dx * 2.0 + ballW * marginScale).ceil().clamp(2, imgW);
      int searchH = (dy * 2.0 + ballH * marginScale).ceil().clamp(2, imgH);

      int searchX = (centerPxX - searchW / 2.0).floor().clamp(0, imgW - searchW);
      int searchY = (centerPxY - searchH / 2.0).floor().clamp(0, imgH - searchH);

      final searchRect = cv.Rect(searchX, searchY, searchW, searchH);
      searchRoi = nextMat.region(searchRect);

      // 2. DUAL-TEMPLATE CONFIDENCE SCAN
      var matchRes = cv.matchTemplate(searchRoi, _template!, cv.TM_CCOEFF_NORMED);
      var (_, maxVal, _, maxLoc) = cv.minMaxLoc(matchRes);
      matchRes.dispose();

      if (maxVal < 0.35 && _anchorTemplate != null) {
         var anchorRes = cv.matchTemplate(searchRoi, _anchorTemplate!, cv.TM_CCOEFF_NORMED);
         var anchorMatch = cv.minMaxLoc(anchorRes);
         anchorRes.dispose();
         if (anchorMatch.$2 > maxVal) {
           maxVal = anchorMatch.$2;
           maxLoc = anchorMatch.$4;
         }
      }

      // 3. RECOVERY & VELOCITY RESET
      if (maxVal < 0.18) { 
        _consecutiveLost++;
        // If lost, keep searching around last position with 0 velocity
        _vx = 0; _vy = 0; 
        nextMat.dispose();
        searchRoi.dispose();
        return null;
      }
      _consecutiveLost = 0;

      double bestX = searchX + maxLoc.x.toDouble();
      double bestY = searchY + maxLoc.y.toDouble();

      // Dynamically preserve the actual updated dimensions of the tracking bounding box 
      // instead of hardcoding _currentBox!.width and height across frames.
      Rect newBox = Rect.fromLTWH(
        (bestX / imgW).clamp(0.0, 1.0),
        (bestY / imgH).clamp(0.0, 1.0),
        _currentBox!.width,
        _currentBox!.height,
      );

      // 4. BOUNCE-FRIENDLY VELOCITY
      double newVx = newBox.left - _currentBox!.left;
      double newVy = newBox.top - _currentBox!.top;

      // We allow massive directional changes (like a bounce) but cap 
      // total displacement to prevent "warping" to other objects.
      double jumpDist = (newBox.center - _currentBox!.center).distance;
      if (jumpDist > 0.35) { // Allow up to 35% of screen travel per frame
        nextMat.dispose();
        searchRoi.dispose();
        return null;
      }

      // 4. VELOCITY UPDATE (With Smoothing)
      // Restore the proven stable dampening factor
      _vx = newVx * 0.7 + _vx * 0.3;
      _vy = newVy * 0.7 + _vy * 0.3;
      _currentBox = newBox;

      // 5. SELECTIVE TEMPLATE UPDATE
      // Only learn new pixels if the match is VERY strong. 
      // This prevents 'learning' the ground during a bounce.
      if (maxVal > 0.6) {
        int templateW = _template!.cols;
        int templateH = _template!.rows;
        int tX = bestX.round().clamp(0, imgW - templateW);
        int tY = bestY.round().clamp(0, imgH - templateH);

        _template?.dispose();
        _template = nextMat.region(cv.Rect(tX, tY, templateW, templateH));
      }

      nextMat.dispose();
      searchRoi.dispose();

      return _currentBox;
    } catch (e) {
      debugPrint("Template Matching Safe Catch: $e");
      try { nextMat?.dispose(); } catch (_) {}
      try { searchRoi?.dispose(); } catch (_) {}
      return null;
    }
  }

  void dispose() {
    try {
      _template?.dispose();
      _anchorTemplate?.dispose();
    } catch (_) {}
  }
}