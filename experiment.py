import json
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    "TrevorJS/clef-flash-mlx-8bit",
    local_dir="models/clef-flash-8bit",
)
sys.path.insert(0, path)

from clef_mlx import load, decide

model = load(path)
result = decide(model, {
    "state": "The production database is unavailable and users cannot sign in.",
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does this incident require immediate attention?"
        }
    }
})
print(json.dumps(result, indent=2))
