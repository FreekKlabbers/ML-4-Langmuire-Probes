from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError


def _upload_one(api: HfApi, repo_id: str, path: str, commit_message: str):
    path = Path(path)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Uploaded {path.name} -> {repo_id}")


def upload(
    repo_id: str,
    weights_path: str,
    config_path: str | None = None,
    extra_paths: list[str] | None = None,
    private: bool = False,
    commit_message: str = "Upload weights",
):
    api = HfApi()

    try:
        api.whoami()
    except Exception as exc:
        raise PermissionError(
            "Hugging Face upload requires authentication. Run 'huggingface-cli login' "
            "or set HF_TOKEN, and make sure the token belongs to the repo owner."
        ) from exc

    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

        _upload_one(api, repo_id, weights_path, commit_message)

        if config_path:
            _upload_one(api, repo_id, config_path, commit_message)

        for path in extra_paths or []:
            _upload_one(api, repo_id, path, commit_message)

        print(f"Done: https://huggingface.co/{repo_id}")
    except HfHubHTTPError as exc:
        if getattr(exc, "response", None) is not None and exc.response.status_code in {
            401,
            403,
        }:
            raise PermissionError(
                f"Authentication failed for repo '{repo_id}'. "
                "Check that you are logged in with a valid token and that the token can write to this user/org."
            ) from exc
        raise
