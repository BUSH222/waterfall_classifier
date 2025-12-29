from __future__ import annotations

import random
from pathlib import Path
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND

from .auth import SESSION_USER_KEY, authenticate, get_current_user, require_user_for_api
from .config import Settings, load_settings
from .db import AnnotationDB
from .labels import load_ground_truth


def _scan_images(dataset_root: Path) -> list[str]:
    """Returns image paths relative to dataset_root.

    Expected dataset layout:
      dataset_root/with_signal/**/*.png
      dataset_root/without_signal/**/*.png
    """

    if not dataset_root.exists():
        return []

    results: list[str] = []
    for folder_name in ("with_signal", "without_signal"):
        folder = dataset_root / folder_name
        if not folder.exists():
            continue

        for p in folder.rglob("*.png"):
            if not p.is_file():
                continue
            results.append(str(p.relative_to(dataset_root).as_posix()))

    results.sort()
    return results


def _derive_ground_truth_from_paths(image_paths: list[str]) -> dict[str, int]:
    gt: dict[str, int] = {}
    for rel in image_paths:
        # rel is like: with_signal/foo.png OR without_signal/bar.png
        first = rel.split("/", 1)[0]
        if first == "with_signal":
            gt[rel] = 1
        elif first == "without_signal":
            gt[rel] = 0
    return gt


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    base_dir = Path(__file__).resolve().parents[1]
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    app = FastAPI(title="Waterfall Manual Classifier")
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

    db = AnnotationDB(settings.sqlite_path)

    images = _scan_images(settings.images_dir)
    if settings.labels_file is not None:
        ground_truth = load_ground_truth(settings.labels_file)
    else:
        # If the dataset is already split into folders, infer labels from that split.
        ground_truth = _derive_ground_truth_from_paths(images)

    app.state.settings = settings
    app.state.db = db
    app.state.ground_truth = ground_truth
    app.state.images = images
    app.state.images_set = set(app.state.images)

    if ground_truth:
        db.refresh_ground_truth(ground_truth=ground_truth)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        db.close()

    @app.get("/", include_in_schema=False)
    def index(request: Request):
        user = get_current_user(request)
        if user:
            return RedirectResponse(url="/classify", status_code=HTTP_302_FOUND)
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page(request: Request):
        settings: Settings = request.app.state.settings
        if not settings.users:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "request": request,
                    "error": "No users configured. Set WF_USERS (e.g. alice:pass,bob:pass).",
                },
            )
        return templates.TemplateResponse(request, "login.html", {"request": request, "error": None})

    @app.post("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_action(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        settings: Settings = request.app.state.settings
        if not settings.users:
            return templates.TemplateResponse(
                request,
                "login.html",
                {"request": request, "error": "No users configured on server."},
                status_code=400,
            )

        if not authenticate(settings.users, username=username, password=password):
            return templates.TemplateResponse(
                request,
                "login.html",
                {"request": request, "error": "Invalid username or password."},
                status_code=401,
            )

        request.session[SESSION_USER_KEY] = username
        return RedirectResponse(url="/classify", status_code=HTTP_302_FOUND)

    @app.post("/logout", include_in_schema=False)
    def logout(request: Request):
        request.session.pop(SESSION_USER_KEY, None)
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    @app.get("/classify", response_class=HTMLResponse, include_in_schema=False)
    def classify_page(request: Request):
        user = get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

        images: list[str] = request.app.state.images
        if not images:
            return templates.TemplateResponse(
                request,
                "done.html",
                {
                    "request": request,
                    "message": f"No images found in {request.app.state.settings.images_dir}",
                },
            )

        db: AnnotationDB = request.app.state.db
        classified = db.user_classified_filenames(user)

        remaining = [name for name in images if name not in classified]
        if not remaining:
            return templates.TemplateResponse(
                request,
                "done.html",
                {
                    "request": request,
                    "message": "All images have been classified for this user. Thank you! 🎉",
                },
            )

        chosen = random.choice(remaining)
        stats = db.user_stats(user)
        total_images = len(images)

        return templates.TemplateResponse(
            request,
            "classify.html",
            {
                "request": request,
                "filename": chosen,
                "total_images": total_images,
                "classified_count": stats["total"],
                "stats": stats,
            },
        )

    def _next_payload(request: Request, *, user: str) -> dict:
        images: list[str] = request.app.state.images
        db: AnnotationDB = request.app.state.db

        classified = db.user_classified_filenames(user)
        remaining = [name for name in images if name not in classified]

        stats = db.user_stats(user)
        stats["total_images"] = len(images)
        stats["remaining"] = len(remaining)
        stats["labels_loaded"] = bool(request.app.state.ground_truth)

        if not remaining:
            return {
                "done": True,
                "filename": None,
                "message": "All images have been classified for this user.",
                "stats": stats,
            }

        return {
            "done": False,
            "filename": random.choice(remaining),
            "message": None,
            "stats": stats,
        }

    @app.post("/classify", include_in_schema=False)
    def classify_action(
        request: Request,
        filename: str = Form(...),
        decision: str = Form(...),
    ):
        user = get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

        if filename not in request.app.state.images_set:
            raise HTTPException(status_code=400, detail="Unknown filename")

        decision_int = 1 if decision == "signal" else 0
        gt_map: dict[str, int] = request.app.state.ground_truth
        gt = gt_map.get(filename)

        db: AnnotationDB = request.app.state.db
        db.upsert_decision(username=user, filename=filename, decision=decision_int, ground_truth=gt)

        return RedirectResponse(url="/classify", status_code=HTTP_302_FOUND)

    @app.get("/api/next")
    def api_next(request: Request, user: str = Depends(require_user_for_api)):
        return _next_payload(request, user=user)

    @app.post("/api/decision")
    def api_decision(
        request: Request,
        filename: str = Form(...),
        decision: str = Form(...),
        user: str = Depends(require_user_for_api),
    ):
        if filename not in request.app.state.images_set:
            raise HTTPException(status_code=400, detail="Unknown filename")

        decision_norm = decision.strip().lower()
        if decision_norm in {"signal", "with_signal", "with-signal", "1", "true", "yes"}:
            decision_int = 1
        elif decision_norm in {"no_signal", "no-signal", "without_signal", "without-signal", "0", "false", "no"}:
            decision_int = 0
        else:
            raise HTTPException(status_code=400, detail="Invalid decision")

        gt_map: dict[str, int] = request.app.state.ground_truth
        gt = gt_map.get(filename)

        db: AnnotationDB = request.app.state.db
        db.upsert_decision(username=user, filename=filename, decision=decision_int, ground_truth=gt)

        return _next_payload(request, user=user)

    @app.post("/api/undo")
    def api_undo(
        request: Request,
        filename: str = Form(...),
        user: str = Depends(require_user_for_api),
    ):
        if filename not in request.app.state.images_set:
            raise HTTPException(status_code=400, detail="Unknown filename")

        db: AnnotationDB = request.app.state.db
        db.delete_decision(username=user, filename=filename)

        # Return the undone image as the next one to show.
        stats = db.user_stats(user)
        stats["total_images"] = len(request.app.state.images)
        stats["remaining"] = stats["total_images"] - stats["total"]
        stats["labels_loaded"] = bool(request.app.state.ground_truth)

        return {
            "done": False,
            "filename": filename,
            "message": "Undid last decision.",
            "stats": stats,
        }

    @app.get("/image/{image_path:path}", name="get_image", include_in_schema=False)
    def get_image(request: Request, image_path: str):
        if image_path not in request.app.state.images_set:
            raise HTTPException(status_code=404, detail="Image not found")

        images_dir: Path = request.app.state.settings.images_dir
        full_path = (images_dir / image_path).resolve()

        # Safety: ensure resolve stays within images_dir
        if images_dir not in full_path.parents and full_path != images_dir:
            raise HTTPException(status_code=400, detail="Invalid path")

        return FileResponse(full_path)

    @app.get("/api/me")
    def api_me(request: Request, user: str = Depends(require_user_for_api)):
        return {"user": user}

    @app.get("/api/stats")
    def api_stats(request: Request, user: str = Depends(require_user_for_api)):
        db: AnnotationDB = request.app.state.db
        stats = db.user_stats(user)
        stats["total_images"] = len(request.app.state.images)
        stats["remaining"] = stats["total_images"] - stats["total"]
        stats["labels_loaded"] = bool(request.app.state.ground_truth)
        return stats

    @app.get("/api/misidentified")
    def api_misidentified(request: Request, limit: int = 500, user: str = Depends(require_user_for_api)):
        db: AnnotationDB = request.app.state.db
        items = db.list_filenames_by_correctness(username=user, correct=0, limit=limit)
        return {
            "count": len(items),
            "items": [
                {
                    "filename": d.filename,
                    "decision": d.decision,
                    "ground_truth": d.ground_truth,
                    "correct": d.correct,
                }
                for d in items
            ],
        }

    @app.get("/api/correct")
    def api_correct(request: Request, limit: int = 500, user: str = Depends(require_user_for_api)):
        db: AnnotationDB = request.app.state.db
        items = db.list_filenames_by_correctness(username=user, correct=1, limit=limit)
        return {
            "count": len(items),
            "items": [
                {
                    "filename": d.filename,
                    "decision": d.decision,
                    "ground_truth": d.ground_truth,
                    "correct": d.correct,
                }
                for d in items
            ],
        }

    return app


app = create_app()
