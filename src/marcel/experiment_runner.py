import json
import time
from pathlib import Path
from typing import Any, Dict

from tqdm.auto import tqdm


def run_experiment(
    pipeline,
    pipeline_runner,
    queries,
    run_path,
    config: Dict[str, Any],
    max_workers=8,  # deprecated, no-op
    **pipeline_args,
):
    run_path = Path(run_path)
    output_json = run_path / "output.json"
    pipeline_json = run_path / "pipeline.json"
    config_json = run_path / "config.json"

    if Path(output_json).exists():
        print(f"{output_json} exists. SKIP.")
        return
    else:
        print("=" * 30, f"Run: {run_path}", "=" * 30)

    pipeline.warm_up()
    pipeline_runner(pipeline, queries[0])  # run one query to warmup the pipeline

    predictions = []
    for query in tqdm(queries):
        start = time.time()
        result = pipeline_runner(pipeline, query)
        end = time.time()
        duration = end - start

        # convert results into canonical evaluation format
        prediction = {
            **query,
            "generated_answer": result["generated_answer"],  # type: ignore
            "contexts": [
                {
                    "content": doc.content,
                    "url": doc.meta["url"],
                    "score": float(doc.score),
                }
                for doc in result["documents"]  # type: ignore
            ],
            "duration": duration,
        }
        predictions.append(prediction)

    run_path.mkdir(exist_ok=True, parents=True)
    with open(output_json, "w") as fout:
        json.dump(predictions, fout)
    with open(pipeline_json, "w") as fout:
        json.dump(pipeline.to_dict(), fout, indent=4)
    with open(config_json, "w") as fout:
        json.dump(config, fout, indent=4)
