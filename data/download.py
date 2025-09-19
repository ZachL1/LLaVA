from huggingface_hub import snapshot_download


######### Models #########
save_dir = "LLaVA-Pretrai"
repo_id = "liuhaotian/LLaVA-Pretrain"
snapshot_download(
  local_dir=save_dir,
  repo_id=repo_id,
  repo_type="dataset",
  local_dir_use_symlinks=False,
  resume_download=True,
)
