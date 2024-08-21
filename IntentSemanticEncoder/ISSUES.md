# evaluate package dependency issue
```bash
The conflict is caused by:
    transformers 4.18.0 depends on huggingface-hub<1.0 and >=0.1.0
    sentence-transformers 2.2.2 depends on huggingface-hub>=0.4.0
    allennlp 2.9.3 depends on huggingface-hub>=0.0.16
    cached-path 1.1.2 depends on huggingface-hub<0.6.0 and >=0.0.12
    evaluate 0.1.2 depends on huggingface-hub>=0.7.0

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict
```

# Inspectiont of multiwoz
schema: `datasets/multiwoz/data/MultiWOZ_2.2/schema.json`
len=8

# Install build wheel error for tokenizers
Do this:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Restart your terminal, and then
```bash
python -m pip install -r requirements.txt
```
Reference: https://stackoverflow.com/questions/69595700/could-not-build-wheels-for-tokenizers-which-is-required-to-install-pyproject-to

# boto3 error credential expires
You can try using `ada`:
https://builderhub.corp.amazon.com/docs/codeartifact/user-guide/authentication.html#install-and-configure-the-ada-cli
And remember to add this:
```bash
export AWS_PROFILE=profile_name
```
Also, write a script to the automatically relaunch the script when error occurs.
Reference: https://stackoverflow.com/questions/35331824/bash-script-to-re-launch-program-in-case-of-failure-error

**UPDATES Sept 22 2023**
The boto3 error was solved by updating client every 10min.
See code here `build_toolkit_with_endpoints/extract_goals.py` at line 81.