from huggingface_hub import snapshot_download


######### Models #########
save_dir = "vicuna-13b-v1.5"
repo_id = "lmsys/vicuna-13b-v1.5"
snapshot_download(
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
)