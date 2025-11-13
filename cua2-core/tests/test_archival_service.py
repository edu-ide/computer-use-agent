"""
Tests for the ArchivalService multiprocessing implementation.
"""

import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from cua2_core.services.archival_service import (
    ArchivalService,
    _compress_folder,
    _process_old_folders,
    _upload_to_huggingface,
    _verify_file_in_repo,
)
from huggingface_hub.utils import HfHubHTTPError


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hf_api():
    """Create a mock HuggingFace API client."""
    mock_api = MagicMock()
    mock_api.upload_file.return_value = None
    return mock_api


@pytest.fixture
def archival_service(temp_data_dir):
    """Create an ArchivalService instance for testing."""
    service = ArchivalService(
        hf_token="test_token",
        hf_dataset_repo="test_user/test_repo",
        data_dir=temp_data_dir,
        archive_interval_minutes=1,  # Short interval for testing
        folder_age_threshold_minutes=1,
    )
    yield service
    # Cleanup - stop the service if running
    if service.is_alive():
        service.stop(timeout=5.0)


class TestArchivalServiceInitialization:
    """Test ArchivalService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}, clear=False):
            # Need to pass the env token explicitly since os.getenv is called at function definition time
            service = ArchivalService(hf_token=os.getenv("HF_TOKEN"))
            assert service.hf_token == "env_token"
            assert service.hf_dataset_repo == "smolagents/cua_traces"
            assert service.data_dir == "data"
            assert service.archive_interval_minutes == 30
            assert service.folder_age_threshold_minutes == 30
            assert service._process is None
            assert not service.is_alive()

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        service = ArchivalService(
            hf_token="custom_token",
            hf_dataset_repo="custom/repo",
            data_dir="/custom/path",
            archive_interval_minutes=60,
            folder_age_threshold_minutes=120,
        )
        assert service.hf_token == "custom_token"
        assert service.hf_dataset_repo == "custom/repo"
        assert service.data_dir == "/custom/path"
        assert service.archive_interval_minutes == 60
        assert service.folder_age_threshold_minutes == 120

    def test_init_multiprocessing_components(self):
        """Test that multiprocessing components are initialized."""
        service = ArchivalService(hf_token="test", hf_dataset_repo="test/test")
        assert service._stop_event is not None
        assert service._manager is not None
        assert service._active_tasks is not None


class TestArchivalServiceLifecycle:
    """Test ArchivalService lifecycle management."""

    def test_start_service(self, archival_service):
        """Test starting the archival service."""
        archival_service.start()

        # Give the process a moment to start
        time.sleep(0.5)

        assert archival_service.is_alive()
        assert archival_service._process is not None
        assert archival_service._process.pid is not None

    def test_start_without_credentials(self, temp_data_dir):
        """Test starting service without credentials logs warning."""
        service = ArchivalService(
            hf_token=None,
            hf_dataset_repo=None,
            data_dir=temp_data_dir,
        )

        service.start()

        # Service should not start
        assert not service.is_alive()
        assert service._process is None

    def test_start_already_running(self, archival_service):
        """Test starting service when already running."""
        archival_service.start()
        time.sleep(0.5)

        pid1 = archival_service._process.pid

        # Try to start again
        archival_service.start()

        # Should be same process
        assert archival_service._process.pid == pid1

    def test_stop_service(self, archival_service):
        """Test stopping the archival service."""
        archival_service.start()
        time.sleep(0.5)
        assert archival_service.is_alive()

        archival_service.stop(timeout=5.0)

        assert not archival_service.is_alive()
        assert archival_service._process is None

    def test_stop_not_running(self, archival_service):
        """Test stopping service when not running."""
        # Should not raise any errors
        archival_service.stop(timeout=1.0)
        assert not archival_service.is_alive()

    def test_stop_with_timeout(self, archival_service):
        """Test stop with timeout and force termination."""
        archival_service.start()
        time.sleep(0.5)

        # Stop with very short timeout to test force kill path
        archival_service.stop(timeout=0.001)

        # Process should be stopped one way or another
        time.sleep(0.5)
        assert not archival_service.is_alive()

    def test_is_alive_returns_false_when_not_started(self, archival_service):
        """Test is_alive returns False when service not started."""
        assert not archival_service.is_alive()


class TestActiveTasksManagement:
    """Test active tasks management."""

    def test_update_active_tasks(self, archival_service):
        """Test updating active tasks."""
        task_ids = {"task-1", "task-2", "task-3"}

        archival_service.update_active_tasks(task_ids)

        # Verify tasks are in shared dict
        for task_id in task_ids:
            assert task_id in archival_service._active_tasks

    def test_update_active_tasks_clears_old(self, archival_service):
        """Test that updating active tasks clears old ones."""
        archival_service.update_active_tasks({"task-1", "task-2"})
        assert "task-1" in archival_service._active_tasks

        archival_service.update_active_tasks({"task-3"})
        assert "task-1" not in archival_service._active_tasks
        assert "task-3" in archival_service._active_tasks

    def test_update_active_tasks_empty_set(self, archival_service):
        """Test updating with empty set."""
        archival_service.update_active_tasks({"task-1"})
        archival_service.update_active_tasks(set())

        assert len(archival_service._active_tasks) == 0


class TestCompressFolder:
    """Test folder compression functionality."""

    def test_compress_folder_success(self, temp_data_dir):
        """Test successful folder compression."""
        # Create a test folder with some files
        test_folder = Path(temp_data_dir) / "trace-test-123-model"
        test_folder.mkdir()
        (test_folder / "file1.txt").write_text("test content 1")
        (test_folder / "file2.txt").write_text("test content 2")

        archive_path = _compress_folder(test_folder)

        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.name == "trace-test-123-model.tar.gz"
        assert archive_path.suffix == ".gz"

        # Verify archive contents
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
            assert "trace-test-123-model/file1.txt" in members
            assert "trace-test-123-model/file2.txt" in members

        # Cleanup
        archive_path.unlink()

    def test_compress_folder_empty_folder(self, temp_data_dir):
        """Test compressing an empty folder."""
        test_folder = Path(temp_data_dir) / "trace-empty-456-model"
        test_folder.mkdir()

        archive_path = _compress_folder(test_folder)

        assert archive_path is not None
        assert archive_path.exists()

        # Cleanup
        archive_path.unlink()

    def test_compress_folder_nonexistent(self, temp_data_dir):
        """Test compressing a nonexistent folder."""
        test_folder = Path(temp_data_dir) / "nonexistent"

        archive_path = _compress_folder(test_folder)

        assert archive_path is None

    def test_compress_folder_with_subdirectories(self, temp_data_dir):
        """Test compressing folder with subdirectories."""
        test_folder = Path(temp_data_dir) / "trace-nested-789-model"
        test_folder.mkdir()
        subdir = test_folder / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")

        archive_path = _compress_folder(test_folder)

        assert archive_path is not None

        # Verify nested structure preserved
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
            assert "trace-nested-789-model/subdir/nested.txt" in members

        # Cleanup
        archive_path.unlink()


class TestUploadToHuggingFace:
    """Test HuggingFace upload functionality."""

    def test_upload_success(self, mock_hf_api, temp_data_dir):
        """Test successful upload to HuggingFace."""
        # Create a test archive
        archive_path = Path(temp_data_dir) / "test-archive.tar.gz"
        archive_path.write_text("test archive content")

        result = _upload_to_huggingface(mock_hf_api, "test/repo", archive_path)

        assert result is True
        mock_hf_api.upload_file.assert_called_once_with(
            path_or_fileobj=str(archive_path),
            path_in_repo="test-archive.tar.gz",
            repo_id="test/repo",
            repo_type="dataset",
        )

    def test_upload_failure(self, mock_hf_api, temp_data_dir):
        """Test upload failure."""
        mock_hf_api.upload_file.side_effect = Exception("Upload failed")

        archive_path = Path(temp_data_dir) / "test-archive.tar.gz"
        archive_path.write_text("test archive content")

        result = _upload_to_huggingface(mock_hf_api, "test/repo", archive_path)

        assert result is False

    def test_upload_nonexistent_file(self, mock_hf_api, temp_data_dir):
        """Test uploading a nonexistent file."""
        archive_path = Path(temp_data_dir) / "nonexistent.tar.gz"

        # Make the mock raise an exception when trying to upload nonexistent file
        mock_hf_api.upload_file.side_effect = FileNotFoundError("File not found")

        result = _upload_to_huggingface(mock_hf_api, "test/repo", archive_path)

        assert result is False


class TestVerifyFileInRepo:
    """Test file verification functionality."""

    @patch("cua2_core.services.archival_service.hf_hub_download")
    def test_verify_success(self, mock_download):
        """Test successful file verification."""
        mock_download.return_value = "/path/to/file"

        result = _verify_file_in_repo("test/repo", "test_token", "test.tar.gz")

        assert result is True
        mock_download.assert_called_once()

    @patch("cua2_core.services.archival_service.hf_hub_download")
    def test_verify_file_not_found(self, mock_download):
        """Test verification when file not found (404)."""
        mock_response = Mock()
        mock_response.status_code = 404
        error = HfHubHTTPError("Not found", response=mock_response)
        mock_download.side_effect = error

        result = _verify_file_in_repo("test/repo", "test_token", "test.tar.gz")

        assert result is False

    @patch("cua2_core.services.archival_service.hf_hub_download")
    def test_verify_other_http_error(self, mock_download):
        """Test verification with other HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        error = HfHubHTTPError("Server error", response=mock_response)
        mock_download.side_effect = error

        result = _verify_file_in_repo("test/repo", "test_token", "test.tar.gz")

        assert result is False

    @patch("cua2_core.services.archival_service.hf_hub_download")
    def test_verify_generic_exception(self, mock_download):
        """Test verification with generic exception."""
        mock_download.side_effect = Exception("Generic error")

        result = _verify_file_in_repo("test/repo", "test_token", "test.tar.gz")

        assert result is False


class TestProcessOldFolders:
    """Test old folder processing logic."""

    def test_process_old_folders_basic(self, temp_data_dir, mock_hf_api):
        """Test processing old folders."""
        # Create an old folder (modify mtime to make it old)
        old_folder = Path(temp_data_dir) / "trace-old123-model"
        old_folder.mkdir()
        (old_folder / "data.json").write_text('{"test": "data"}')

        # Make it old by modifying mtime
        old_time = time.time() - 3600  # 1 hour ago
        os.utime(old_folder, (old_time, old_time))

        active_tasks = {}

        with patch(
            "cua2_core.services.archival_service._verify_file_in_repo",
            return_value=True,
        ):
            _process_old_folders(
                data_dir=temp_data_dir,
                folder_age_threshold_minutes=1,
                active_tasks=active_tasks,
                hf_api=mock_hf_api,
                hf_dataset_repo="test/repo",
                hf_token="test_token",
            )

        # Folder should be deleted after successful archival
        assert not old_folder.exists()

        # Upload should have been called
        assert mock_hf_api.upload_file.called

    def test_process_folders_skips_active_tasks(self, temp_data_dir, mock_hf_api):
        """Test that active tasks are skipped."""
        # Create a folder for an active task
        active_folder = Path(temp_data_dir) / "trace-active456-model"
        active_folder.mkdir()
        (active_folder / "data.json").write_text('{"test": "data"}')

        # Make it old
        old_time = time.time() - 3600
        os.utime(active_folder, (old_time, old_time))

        # Mark as active
        active_tasks = {"active456": True}

        _process_old_folders(
            data_dir=temp_data_dir,
            folder_age_threshold_minutes=1,
            active_tasks=active_tasks,
            hf_api=mock_hf_api,
            hf_dataset_repo="test/repo",
            hf_token="test_token",
        )

        # Folder should still exist (not archived)
        assert active_folder.exists()

        # Upload should not have been called
        assert not mock_hf_api.upload_file.called

    def test_process_folders_skips_recent(self, temp_data_dir, mock_hf_api):
        """Test that recent folders are skipped."""
        # Create a recent folder
        recent_folder = Path(temp_data_dir) / "trace-recent789-model"
        recent_folder.mkdir()
        (recent_folder / "data.json").write_text('{"test": "data"}')

        # Folder is fresh (current time)

        active_tasks = {}

        _process_old_folders(
            data_dir=temp_data_dir,
            folder_age_threshold_minutes=60,  # 60 minutes threshold
            active_tasks=active_tasks,
            hf_api=mock_hf_api,
            hf_dataset_repo="test/repo",
            hf_token="test_token",
        )

        # Folder should still exist (too recent)
        assert recent_folder.exists()

        # Upload should not have been called
        assert not mock_hf_api.upload_file.called

    def test_process_folders_keeps_on_verification_failure(
        self, temp_data_dir, mock_hf_api
    ):
        """Test that folders are kept if verification fails."""
        old_folder = Path(temp_data_dir) / "trace-verify-fail-model"
        old_folder.mkdir()
        (old_folder / "data.json").write_text('{"test": "data"}')

        old_time = time.time() - 3600
        os.utime(old_folder, (old_time, old_time))

        active_tasks = {}

        # Mock verification to fail
        with patch(
            "cua2_core.services.archival_service._verify_file_in_repo",
            return_value=False,
        ):
            _process_old_folders(
                data_dir=temp_data_dir,
                folder_age_threshold_minutes=1,
                active_tasks=active_tasks,
                hf_api=mock_hf_api,
                hf_dataset_repo="test/repo",
                hf_token="test_token",
            )

        # Folder should still exist (verification failed)
        assert old_folder.exists()

    def test_process_folders_handles_nonexistent_dir(self, mock_hf_api):
        """Test handling of nonexistent data directory."""
        # Should not raise exception
        _process_old_folders(
            data_dir="/nonexistent/path",
            folder_age_threshold_minutes=1,
            active_tasks={},
            hf_api=mock_hf_api,
            hf_dataset_repo="test/repo",
            hf_token="test_token",
        )

        # No uploads should occur
        assert not mock_hf_api.upload_file.called

    def test_process_folders_handles_bad_folder_names(self, temp_data_dir, mock_hf_api):
        """Test handling of folders with unexpected name format."""
        # Create folder with bad name format
        bad_folder = Path(temp_data_dir) / "trace-invalid"  # Missing parts
        bad_folder.mkdir()

        old_time = time.time() - 3600
        os.utime(bad_folder, (old_time, old_time))

        # Should not raise exception
        _process_old_folders(
            data_dir=temp_data_dir,
            folder_age_threshold_minutes=1,
            active_tasks={},
            hf_api=mock_hf_api,
            hf_dataset_repo="test/repo",
            hf_token="test_token",
        )

        # Folder should still exist (invalid name)
        assert bad_folder.exists()


class TestIntegration:
    """Integration tests for the complete archival workflow."""

    @pytest.mark.slow
    def test_full_archival_workflow(self, temp_data_dir):
        """Test the complete archival workflow end-to-end."""
        # Create service
        service = ArchivalService(
            hf_token="test_token",
            hf_dataset_repo="test/repo",
            data_dir=temp_data_dir,
            archive_interval_minutes=1,
            folder_age_threshold_minutes=1,
        )

        # Create test folder
        test_folder = Path(temp_data_dir) / "trace-integration-test-model"
        test_folder.mkdir()
        (test_folder / "test.json").write_text('{"test": "data"}')

        # Make it old
        old_time = time.time() - 3600
        os.utime(test_folder, (old_time, old_time))

        # Start service (mocked to prevent actual HF upload)
        with (
            patch(
                "cua2_core.services.archival_service._upload_to_huggingface",
                return_value=True,
            ),
            patch(
                "cua2_core.services.archival_service._verify_file_in_repo",
                return_value=True,
            ),
        ):
            service.start()
            time.sleep(2)  # Wait for at least one cycle

            service.stop(timeout=5.0)

        assert not service.is_alive()

    def test_service_survives_worker_errors(self, temp_data_dir):
        """Test that service continues running despite worker errors."""
        service = ArchivalService(
            hf_token="test_token",
            hf_dataset_repo="test/repo",
            data_dir=temp_data_dir,
            archive_interval_minutes=1,
            folder_age_threshold_minutes=1,
        )

        # Mock to raise exception
        with patch(
            "cua2_core.services.archival_service._process_old_folders",
            side_effect=Exception("Test error"),
        ):
            service.start()
            time.sleep(2)

            # Service should still be alive
            assert service.is_alive()

            service.stop(timeout=5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
