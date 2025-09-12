This directory should contain the local Qwen2.5-7B tokenizer files to avoid online access:

Required files (place here):
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json
- added_tokens.json (optional)

After placing these files, the local smoke test and training can load the tokenizer via
`vibevoice_no/vibevoice_no_7b/preprocessor_config.json` which points to this directory.


