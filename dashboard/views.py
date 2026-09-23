"""
Views and API controllers for the rPPG dashboard web application.
"""

import os
import time
import json
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from dashboard.streamer import RPPGStreamManager
from dashboard.ground_truth import parse_ground_truth


def index(request):
    """Renders the main rPPG clinical and research dashboard."""
    return render(request, "dashboard/index.html")


def video_stream(request):
    """MJPEG streaming endpoint for the live processed video feed."""
    stream_mgr = RPPGStreamManager.get_instance()

    def frame_generator():
        idle_count = 0
        while stream_mgr.is_running or idle_count < 20:
            jpeg_bytes = stream_mgr.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
                idle_count = 0
            else:
                idle_count += 1
            time.sleep(0.033)

    return StreamingHttpResponse(
        frame_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


def api_metrics(request):
    """Returns live telemetry, latest pulse sample, and running evaluation metrics."""
    stream_mgr = RPPGStreamManager.get_instance()
    telemetry = stream_mgr.get_telemetry()
    return JsonResponse(telemetry)


@csrf_exempt
def api_upload(request):
    """Handles upload of video file and optional ground truth reference file."""
    if request.method != "POST":
        return JsonResponse({"error": "POST method required."}, status=405)

    video_file = request.FILES.get("video")
    gt_file = request.FILES.get("ground_truth")

    if not video_file:
        return JsonResponse({"error": "No video file provided."}, status=400)

    # Ensure upload directory exists
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_path = upload_dir / video_file.name
    with open(video_path, "wb+") as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    gt_data = None
    gt_path = None
    if gt_file:
        gt_path = upload_dir / gt_file.name
        with open(gt_path, "wb+") as destination:
            for chunk in gt_file.chunks():
                destination.write(chunk)

        try:
            gt_data = parse_ground_truth(str(gt_path))
        except Exception as e:
            return JsonResponse({
                "success": False,
                "error": f"Failed to parse ground truth file: {str(e)}"
            }, status=400)

    return JsonResponse({
        "success": True,
        "video_path": str(video_path),
        "video_name": video_file.name,
        "has_ground_truth": gt_data is not None,
        "gt_data": gt_data,
    })


def get_available_datasets():
    """Scans the local data/ folder and returns all discoverable videos & ground truth files."""
    data_dir = Path(settings.BASE_DIR) / "data"
    video_exts = (".avi", ".mp4", ".mov", ".mkv")
    gt_exts = (".txt", ".csv", ".xmp")

    datasets = []
    if data_dir.exists():
        for item in sorted(os.listdir(data_dir)):
            p = data_dir / item
            if p.is_dir():
                vids = [f for f in os.listdir(p) if f.lower().endswith(video_exts)]
                gts = [f for f in os.listdir(p) if f.lower().endswith(gt_exts) and not f.startswith("evaluation_")]
                if vids:
                    for v in sorted(vids):
                        vp = p / v
                        size_mb = os.path.getsize(vp) / (1024 * 1024)
                        size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1000 else f"{size_mb:.1f} MB"

                        gt_file = None
                        if gts:
                            pref = [g for g in gts if "ground" in g.lower() or "gtdump" in g.lower() or "gt" in g.lower()]
                            gt_file = pref[0] if pref else gts[0]

                        label_gt = f" + GT: {gt_file}" if gt_file else " (no GT)"
                        datasets.append({
                            "id": f"{item}/{v}",
                            "folder": item,
                            "display_name": f"{item} ({size_str}){label_gt}",
                            "video_file": v,
                            "video_path": str(vp),
                            "video_size": size_str,
                            "has_gt": gt_file is not None,
                            "gt_file": gt_file,
                            "gt_path": str(p / gt_file) if gt_file else None,
                        })
            elif p.is_file() and p.name.lower().endswith(video_exts):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1000 else f"{size_mb:.1f} MB"
                datasets.append({
                    "id": p.name,
                    "folder": "data",
                    "display_name": f"{p.name} ({size_str}, no GT)",
                    "video_file": p.name,
                    "video_path": str(p),
                    "video_size": size_str,
                    "has_gt": False,
                    "gt_file": None,
                    "gt_path": None,
                })
    return datasets


def api_datasets(request):
    """API endpoint to list local datasets found inside data/ folder."""
    return JsonResponse({
        "success": True,
        "datasets": get_available_datasets()
    })


@csrf_exempt
def api_control(request):
    """Controls stream operations (start_webcam, start_video, pause, stop, load_sample, load_dataset)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST method required."}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        data = {}

    action = data.get("action", "")
    stream_mgr = RPPGStreamManager.get_instance()

    if action == "start_webcam":
        cam_idx = data.get("camera_index", 0)
        stream_mgr.start_webcam(camera_index=cam_idx)
        return JsonResponse({"status": "started_webcam"})

    elif action == "start_video":
        video_path = data.get("video_path")
        gt_times = data.get("gt_times", [])
        gt_hr = data.get("gt_hr", [])

        if not video_path:
            return JsonResponse({"error": "No video path specified."}, status=400)

        resolved_path = video_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.join(settings.BASE_DIR, video_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                return JsonResponse({"error": f"Video file not found: {video_path}"}, status=400)

        stream_mgr.start_video(resolved_path, gt_times=gt_times, gt_hr=gt_hr)
        return JsonResponse({"status": "started_video"})

    elif action == "load_dataset":
        dataset_id = data.get("dataset_id")
        datasets = get_available_datasets()
        target = next((d for d in datasets if d["id"] == dataset_id), None)
        if not target:
            return JsonResponse({"success": False, "error": f"Dataset '{dataset_id}' not found."}, status=404)

        video_path = target["video_path"]
        gt_path = target.get("gt_path")

        if not os.path.exists(video_path):
            return JsonResponse({"success": False, "error": f"Video not found at {video_path}"}, status=404)

        gt_data = None
        if gt_path and os.path.exists(gt_path):
            try:
                gt_data = parse_ground_truth(gt_path)
            except Exception as e:
                pass

        return JsonResponse({
            "success": True,
            "video_path": video_path,
            "video_name": f"data/{target['id']}",
            "has_ground_truth": gt_data is not None,
            "gt_data": gt_data,
        })

    elif action == "load_sample":
        # Convenient helper to load local UBFC sample (subject10)
        sample_vid = os.path.join(settings.BASE_DIR, "data", "subject10", "vid.avi")
        sample_gt = os.path.join(settings.BASE_DIR, "data", "subject10", "ground_truth.txt")

        if not os.path.exists(sample_vid):
            return JsonResponse({"error": "Sample video not found at data/subject10/vid.avi"}, status=404)

        gt_data = None
        if os.path.exists(sample_gt):
            try:
                gt_data = parse_ground_truth(sample_gt)
            except Exception as e:
                pass

        return JsonResponse({
            "success": True,
            "video_path": sample_vid,
            "video_name": "data/subject10/vid.avi",
            "has_ground_truth": gt_data is not None,
            "gt_data": gt_data,
        })

    elif action == "pause":
        stream_mgr.pause()
        return JsonResponse({"status": "toggled_pause"})

    elif action == "stop":
        stream_mgr.stop()
        return JsonResponse({"status": "stopped"})

    else:
        return JsonResponse({"error": f"Unknown action '{action}'"}, status=400)
