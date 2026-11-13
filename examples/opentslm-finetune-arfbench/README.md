# Fine-tune OpenTSLM on ARFBench with Amazon SageMaker AI

This example fine-tunes the pretrained OpenTSLM-Flamingo model from [Aionic Labs](https://www.aioniclabs.ai/) on
[ARFBench](https://www.datadoghq.com/blog/ai/introducing-arfbench/) as a SageMaker training job, deploys it to a
real-time endpoint, and asks it a question about a time series. OpenTSLM-Flamingo combines the Gemma 3 4B language
model with a time-series encoder and answers questions about numerical time series in natural language; the
pretrained model is also in Amazon SageMaker JumpStart as **OpenTSLM-Flamingo Gemma 3 4B**. ARFBench, multiple-choice
questions about anomalies in Datadog's incident telemetry, serves as an example task: to adapt the model to your
domain, use your own signals, questions and reviewed answers.

For the walkthrough, see the AWS blog post <!-- TODO: blog URL -->.

## Files

| File | Contents |
|---|---|
| `opentslm-finetune-arfbench.ipynb` | Set up, fine-tune, deploy, ask a question, clean up |
| `scripts/train.py` | The training job's entry point |
| `scripts/utils.py` | The model, its fine-tuning code and the endpoint's request handler |
| `scripts/arfbench-split.json` | The training, validation and test sets |
| `scripts/requirements.txt` | The packages of the training job |

## What the notebook does

1. **Set up:** uses the SageMaker default bucket and the Studio execution role. Outside Studio, it creates the role
   `OpenTSLMSageMakerRole` with `AmazonSageMakerFullAccess` once.
2. **Fine-tune:** `ModelTrainer` runs `scripts/train.py` on one `ml.g7e.2xlarge` instance. The job loads ARFBench
   through [TimeNet](https://docs.timenet.ai/), splits it by incident into training, validation and test sets (about
   70%, 10% and 20%), and fine-tunes for 3 epochs while most of Gemma 3 stays frozen. It saves the model to S3 in the
   layout of a Hugging Face repository, with `metrics.json`: the test accuracy before and after fine-tuning, also per
   question category. `BASE_MODEL` sets the pretrained checkpoint (`hf://ORG/REPO` or a local directory).
3. **Deploy:** serves the fine-tuned model on one `ml.g6.2xlarge` instance with the SageMaker PyTorch 2.6 inference
   image.
4. **Ask a question** and **clean up**: deletes the endpoint, its configuration and the model. The model files, the
   training job's files, the MLflow runs and the role stay in your account; delete them when you no longer need them.

## Prerequisites

- AWS credentials for the account and Region, in SageMaker Studio or on your machine. Outside Studio, the first run
  needs permission to create an IAM role.
- A Region with `ml.g7e.2xlarge` for training and `ml.g6.2xlarge` for endpoints, and quotas of at least 1 for
  "ml.g7e.2xlarge for training job usage" and "ml.g6.2xlarge for endpoint usage" in
  [Service Quotas](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas).
  `ml.g7e.4xlarge` and `ml.g7e.8xlarge` have the same GPU if `ml.g7e.2xlarge` has no capacity.
- Internet access from the training job (Hugging Face, TimeNet) and the endpoint (PyPI).

## Track the training with MLflow (optional)

With `MLFLOW_APP = "default"`, the training job logs its parameters, the loss of each epoch and the test results to
the default [MLflow App](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup.html) of your Studio domain
or account, in the experiment `opentslm-arfbench`. Set an app ARN to use another app, or `None` to turn it off. The
execution role needs `sagemaker:CallMlflowAppApi`, which `AmazonSageMakerFullAccess` includes, and `mlflow-skinny` in
`scripts/requirements.txt` should match the MLflow version of the app.

## Request format

Up to 4 instances per request, each with a question, up to 32 series and optional `context` lines; `max_new_tokens`
from 1 to 512. `None` marks a missing value. The response holds one answer per instance in `generated_text`.

```json
{
  "instances": [{
    "question": "Is there an anomaly in this series? If so, when does it start and end?",
    "series": [{"values": [21, 20, 22, 19, 78, 85, 82, 22, 20, 21],
                "name": "CPU utilization (%)",
                "text": "CPU utilization (%), 1 sample/minute:"}],
    "context": ["The series covers 10:00 to 10:09 on a web server."]
  }],
  "parameters": {"max_new_tokens": 128}
}
```

## License

MIT-0, see the repository [LICENSE](../../LICENSE). `scripts/utils.py` contains Flamingo layers adapted from
[OpenFlamingo](https://github.com/mlfoundations/open_flamingo) and
[flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) (both MIT); their notices are in its docstring. The
pretrained model weights have their own license: see the model card of the checkpoint you use.
