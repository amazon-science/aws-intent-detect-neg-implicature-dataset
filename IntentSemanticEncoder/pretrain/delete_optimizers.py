# walk through the results dir and delete all those `optimizer.pt`

import os

# https://stackoverflow.com/questions/1724693/find-a-file-in-python
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

target_files = find_all('optimizer.pt', 'results')
# check before delete
print(target_files)

for f in target_files:
    os.remove(f)