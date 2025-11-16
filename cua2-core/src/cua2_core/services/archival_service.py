"""
Service for automatic data archival to HuggingFace datasets.

This service runs in a dedicated process that periodically:
1. Scans for old trace data folders
2. Compresses them into tar.gz archives
3. Uploads to HuggingFace dataset repository
4. Verifies successful upload
5. Deletes local files only after verification
"""

import logging
import multiprocessing
import multiprocessing.synchronize
import os
import shutil
import signal
import tarfile
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# Configure logging for the process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ArchivalService:
    """Service for handling automatic data archival to HuggingFace in a dedicated process"""

    def __init__(
        self,
        hf_token: str | None = os.getenv("HF_TOKEN"),
        hf_dataset_repo: str | None = "smolagents/cua_traces",
        data_dir: str = "data",
        archive_interval_minutes: int = 30,
        folder_age_threshold_minutes: int = 30,
    ):
        """
        Initialize the archival service.

        Args:
            hf_token: HuggingFace API token
            hf_dataset_repo: HuggingFace dataset repository ID (e.g., "username/dataset-name")
            data_dir: Directory containing trace data folders
            archive_interval_minutes: How often to check for old folders
            folder_age_threshold_minutes: Minimum age before archival
        """
        self.hf_token = hf_token
        self.hf_dataset_repo = hf_dataset_repo
        self.data_dir = data_dir
        self.archive_interval_minutes = archive_interval_minutes
        self.folder_age_threshold_minutes = folder_age_threshold_minutes

        # Multiprocessing components
        self._process: multiprocessing.Process | None = None
        self._stop_event: multiprocessing.synchronize.Event = multiprocessing.Event()
        self._manager = multiprocessing.Manager()
        self._active_tasks: Any = self._manager.dict()  # DictProxy type

    def start(self):
        """Start the archival service in a dedicated process."""
        if self._process and self._process.is_alive():
            logger.warning("Archival service is already running")
            return

        if not self.hf_token or not self.hf_dataset_repo:
            logger.warning(
                "HuggingFace credentials or dataset repo not configured. Data archival disabled."
            )
            return

        self._stop_event.clear()
        self._process = multiprocessing.Process(
            target=_archival_worker_process,
            args=(
                self.hf_token,
                self.hf_dataset_repo,
                self.data_dir,
                self.archive_interval_minutes,
                self.folder_age_threshold_minutes,
                self._stop_event,
                self._active_tasks,
            ),
            daemon=True,
            name="ArchivalWorker",
        )
        self._process.start()
        logger.info(
            f"Started archival service in process {self._process.pid}. "
            f"Checking every {self.archive_interval_minutes} minutes."
        )

    def stop(self, timeout: float = 10.0):
        """
        Stop the archival service process.

        Args:
            timeout: Maximum time to wait for process to terminate (seconds)
        """
        if not self._process or not self._process.is_alive():
            return

        logger.info(f"Stopping archival service (PID: {self._process.pid})...")
        self._stop_event.set()

        self._process.join(timeout=timeout)

        if self._process.is_alive():
            logger.warning("Archival process did not stop gracefully, terminating...")
            self._process.terminate()
            self._process.join(timeout=2.0)

            if self._process.is_alive():
                logger.error("Force killing archival process...")
                self._process.kill()
                self._process.join()

        logger.info("Archival service stopped")
        self._process = None

    def update_active_tasks(self, active_task_ids: set[str]):
        """
        Update the set of active task IDs.
        The archival process will skip folders for these tasks.

        Args:
            active_task_ids: Set of currently active trace IDs
        """
        # Clear and update the shared dict
        self._active_tasks.clear()
        for task_id in active_task_ids:
            self._active_tasks[task_id] = True

    def is_alive(self) -> bool:
        """Check if the archival process is running."""
        return self._process is not None and self._process.is_alive()


def _archival_worker_process(
    hf_token: str,
    hf_dataset_repo: str,
    data_dir: str,
    archive_interval_minutes: int,
    folder_age_threshold_minutes: int,
    stop_event: multiprocessing.synchronize.Event,
    active_tasks: Any,
):
    """
    Worker process that performs the archival operations.
    Runs in a separate process from the main application.

    Args:
        hf_token: HuggingFace API token
        hf_dataset_repo: HuggingFace dataset repository
        data_dir: Data directory path
        archive_interval_minutes: Check interval
        folder_age_threshold_minutes: Folder age threshold
        stop_event: Event to signal process shutdown
        active_tasks: Shared dict of active task IDs
    """

    def signal_handler(signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, stopping archival worker...")
        stop_event.set()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize HuggingFace API in this process
    hf_api = HfApi(token=hf_token)

    logger.info(
        f"Archival worker started (PID: {os.getpid()}). "
        f"Checking every {archive_interval_minutes} minutes."
    )

    # Main worker loop
    while not stop_event.is_set():
        try:
            # Sleep in small intervals to be responsive to stop_event
            for _ in range(archive_interval_minutes * 60):
                if stop_event.is_set():
                    break
                time.sleep(1)

            if stop_event.is_set():
                break

            logger.info("Starting data archival check...")
            _process_old_folders(
                data_dir=data_dir,
                folder_age_threshold_minutes=folder_age_threshold_minutes,
                active_tasks=active_tasks,
                hf_api=hf_api,
                hf_dataset_repo=hf_dataset_repo,
                hf_token=hf_token,
            )

        except Exception as e:
            logger.error(f"Error in archival worker: {e}", exc_info=True)
            # Continue running despite errors

    logger.info("Archival worker shutting down gracefully")


def _process_old_folders(
    data_dir: str,
    folder_age_threshold_minutes: int,
    active_tasks: Any,
    hf_api: HfApi,
    hf_dataset_repo: str,
    hf_token: str,
):
    """
    Process and archive folders older than the threshold.
    Runs in the archival worker process.
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist")
        return

    data_path = Path(data_dir)
    current_time = time.time()
    threshold_seconds = folder_age_threshold_minutes * 60

    # Get all trace folders
    try:
        trace_folders = [
            f for f in data_path.iterdir() if f.is_dir() and f.name.startswith("trace-")
        ]
    except Exception as e:
        logger.error(f"Error listing data directory: {e}", exc_info=True)
        return

    for folder in trace_folders:
        try:
            # Check if folder is old enough and not currently active
            folder_mtime = folder.stat().st_mtime
            folder_age_seconds = current_time - folder_mtime

            # Extract trace_id from folder name (format: trace-{uuid}-{model_name})
            folder_name = folder.name
            parts = folder_name.split("-", 2)  # Split into ['trace', uuid, model_name]
            if len(parts) < 2:
                logger.warning(f"Unexpected folder name format: {folder_name}")
                continue

            trace_id = parts[1]

            # Skip if folder is still being used by an active task
            if trace_id in active_tasks:
                logger.debug(f"Skipping active task folder: {folder_name}")
                continue

            # Check if folder is old enough
            if folder_age_seconds < threshold_seconds:
                logger.debug(
                    f"Folder {folder_name} is not old enough ({folder_age_seconds / 60:.1f} minutes)"
                )
                continue

            logger.info(
                f"Processing old folder: {folder_name} (age: {folder_age_seconds / 60:.1f} minutes)"
            )

            # Compress the folder
            archive_path = _compress_folder(folder)

            if not archive_path:
                logger.error(f"Failed to compress folder: {folder_name}")
                continue

            # Upload to HuggingFace
            uploaded = _upload_to_huggingface(hf_api, hf_dataset_repo, archive_path)

            if not uploaded:
                logger.error(f"Failed to upload archive: {archive_path.name}")
                # Clean up the local archive file
                archive_path.unlink(missing_ok=True)
                continue

            # Verify the file exists in the repo
            verified = _verify_file_in_repo(
                hf_dataset_repo, hf_token, archive_path.name
            )

            if verified:
                logger.info(
                    f"Successfully verified {archive_path.name} in HuggingFace repo"
                )

                # Delete the local folder (check if it still exists to avoid race conditions)
                if folder.exists():
                    shutil.rmtree(folder)
                    logger.info(f"Deleted local folder: {folder_name}")
                else:
                    logger.warning(f"Folder {folder_name} already deleted, skipping")

                # Delete the local archive
                archive_path.unlink(missing_ok=True)
                logger.info(f"Deleted local archive: {archive_path.name}")
            else:
                logger.error(
                    f"Could not verify {archive_path.name} in repo. Keeping local files."
                )
                # Keep both the folder and archive for safety

        except Exception as e:
            logger.error(f"Error processing folder {folder.name}: {e}", exc_info=True)


def _compress_folder(folder_path: Path) -> Path | None:
    """
    Compress a folder into a tar.gz archive.

    Args:
        folder_path: Path to the folder to compress

    Returns:
        Path to the created archive file, or None if failed
    """
    try:
        archive_name = f"{folder_path.name}.tar.gz"
        archive_path = folder_path.parent / archive_name

        logger.info(f"Compressing {folder_path.name} to {archive_name}")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)

        archive_size = archive_path.stat().st_size
        logger.info(
            f"Created archive {archive_name} ({archive_size / 1024 / 1024:.2f} MB)"
        )

        return archive_path

    except Exception as e:
        logger.error(f"Error compressing folder {folder_path}: {e}", exc_info=True)
        return None


def _upload_to_huggingface(
    hf_api: HfApi, hf_dataset_repo: str, archive_path: Path
) -> bool:
    """
    Upload an archive file to HuggingFace dataset repository.

    Args:
        hf_api: HuggingFace API client
        hf_dataset_repo: HuggingFace dataset repository ID
        archive_path: Path to the archive file

    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        logger.info(
            f"Uploading {archive_path.name} to HuggingFace repo {hf_dataset_repo}"
        )

        hf_api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=archive_path.name,
            repo_id=hf_dataset_repo,
            repo_type="dataset",
        )

        logger.info(f"Successfully uploaded {archive_path.name} to HuggingFace")
        return True

    except Exception as e:
        logger.error(
            f"Error uploading {archive_path.name} to HuggingFace: {e}", exc_info=True
        )
        return False


def _verify_file_in_repo(hf_dataset_repo: str, hf_token: str, filename: str) -> bool:
    """
    Verify that a file exists in the HuggingFace repository.

    Args:
        hf_dataset_repo: HuggingFace dataset repository ID
        hf_token: HuggingFace API token
        filename: Name of the file to verify

    Returns:
        True if file exists in repo, False otherwise
    """
    try:
        logger.info(f"Verifying {filename} exists in HuggingFace repo")

        # Try to get file info - this will raise an error if file doesn't exist
        hf_hub_download(
            repo_id=hf_dataset_repo,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
        )

        logger.info(f"Verified {filename} exists in repo")
        return True

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"File {filename} not found in repo (404)")
        else:
            logger.error(f"HTTP error verifying file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying {filename} in repo: {e}", exc_info=True)
        return False
