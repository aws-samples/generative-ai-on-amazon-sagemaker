# Fine-tune OpenTSLM on ARFBench with Amazon SageMaker AI

This example fine-tunes the pretrained OpenTSLM-Flamingo model from [Aionic Labs](https://www.aioniclabs.ai/) on
[ARFBench](https://www.datadoghq.com/blog/ai/introducing-arfbench/) as a SageMaker training job, deploys the
fine-tuned model to a SageMaker real-time endpoint, and asks it a question about a time series.

OpenTSLM-Flamingo is a time-series language model: it combines the Gemma 3 4B language model with a time-series
encoder and cross-attention layers, and answers questions about numerical time series in natural language. The
pretrained model is also available in Amazon SageMaker JumpStart as **OpenTSLM-Flamingo Gemma 3 4B**.

ARFBench is a public benchmark of multiple-choice questions about anomalies in Datadog's software-incident
telemetry. Here it serves as an example task. To adapt the model to your own domain, use your own signals,
questions and reviewed answers.

For the walkthrough, see the AWS blog post <!-- TODO: blog URL -->.

## Files

| File | Contents |
|---|---|
| `opentslm-finetune-arfbench.ipynb` | Set up, fine-tune on ARFBench, deploy the fine-tuned model, ask a question, clean up |
| `scripts/train.py` | The training job's entry point |
| `scripts/utils.py` | The OpenTSLM model, its fine-tuning code and the endpoint's request handler. The training job and the endpoint both import it |
| `scripts/arfbench-split.json` | The assignment of each ARFBench question to the training, validation or test set |
| `scripts/requirements.txt` | The packages that the training job installs |

## What the notebook does

1. **Set up:** finds or creates the SageMaker default bucket (`sagemaker-<region>-<account ID>`) and an execution
   role. In SageMaker Studio, it uses the Studio execution role. Elsewhere, it creates the role
   `OpenTSLMSageMakerRole` once, with the AWS managed policy `AmazonSageMakerFullAccess`.
2. **Fine-tune:** `ModelTrainer` runs `scripts/train.py` as a SageMaker training job on one `ml.g7e.2xlarge`
   instance, with the SageMaker PyTorch 2.9 training image. The job loads ARFBench through
   [TimeNet](https://docs.timenet.ai/) and splits the questions into training, validation and test sets (about 70%,
   10% and 20%) as listed in `scripts/arfbench-split.json`. All questions about one incident are in the same set, and each
   set has about the same mix of question categories and answers. The job trains the time-series encoder, the
   perceiver resampler, the cross-attention layers and the language model's token embeddings for 3 epochs, uses the
   validation set to choose the best epoch, and keeps the rest of the Gemma 3 language model frozen. It saves the
   fine-tuned model to Amazon S3 in the layout of a Hugging Face model repository, with `metrics.json`: the test
   accuracy before and after fine-tuning, and the accuracy and balanced accuracy of each question category.
   Optionally, it logs the parameters, the loss of each epoch and the test results to an MLflow App (see
   [Track the training with MLflow](#track-the-training-with-mlflow)).
3. **Deploy:** uploads `scripts/utils.py`, a one-line `inference.py` and a `requirements.txt` next to the fine-tuned model
   files, and deploys them to a real-time endpoint on one `ml.g6.2xlarge` instance with the SageMaker PyTorch 2.6
   inference image.
4. **Ask a question:** sends one hour of synthetic CPU utilization data with a question to the endpoint.
5. **Clean up:** deletes the endpoint, its configuration and the SageMaker model.

The pretrained model is set by `BASE_MODEL` in the fine-tuning cell, as a Hugging Face repository
(`hf://ORG/REPO`) or a local model directory. The repository can get newer checkpoints over time, so set it to the
checkpoint that fits your use case.

## Prerequisites

- An AWS account and AWS credentials for the account and Region to use, for example in SageMaker Studio or on your
  machine after `aws sso login`, with `AWS_PROFILE` and `AWS_DEFAULT_REGION` set. Outside SageMaker Studio, your
  credentials need permission to create IAM roles the first time you run the notebook.
- A Jupyter environment. The notebook itself does not need a GPU.
- A Region that offers both `ml.g7e.2xlarge` for training and `ml.g6.2xlarge` for endpoints.
- Service quotas of at least 1 for "ml.g7e.2xlarge for training job usage" and "ml.g6.2xlarge for endpoint usage".
  New accounts often have a quota of 0 for both. Request an increase in
  [Service Quotas](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas) before you start.
  `ml.g7e.4xlarge` and `ml.g7e.8xlarge` have the same GPU as `ml.g7e.2xlarge`: use one of them if the training job
  waits for capacity.
- Internet access from the training job and the endpoint. The training job downloads the pretrained model from
  Hugging Face and ARFBench from TimeNet, and the endpoint installs Transformers from PyPI when it starts.

## Track the training with MLflow

The training job can log to an [MLflow App](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup.html)
of Amazon SageMaker AI. `MLFLOW_APP` in the notebook chooses the app:

- `"default"` (the default): the default MLflow App of your SageMaker Studio domain, or else the default MLflow App
  of your account. Without one, the job trains without MLflow.
- An MLflow App ARN: that app.
- `None`: no MLflow.

The job logs to the experiment `opentslm-arfbench`, with one run per training job. The execution role needs
permission to call the app (`sagemaker:CallMlflowAppApi`, which `AmazonSageMakerFullAccess` includes). Set the
`mlflow-skinny` version in `scripts/requirements.txt` to the MLflow version of your app.

## Request format

A request holds up to 4 instances. Each instance has a question, up to 32 series, and optional `context` lines,
which go between the series and the question. A series has its `values` (`None` marks a missing value), a short
`name`, and a `text` that describes it. `max_new_tokens` accepts values
from 1 to 512. The response holds one answer per instance in `generated_text`.

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

## Clean up

The last cell deletes the endpoint, its configuration and the SageMaker model, also after a failed deployment. The
fine-tuned model files and `metrics.json` stay under `FINETUNED_PREFIX` in your bucket, the training job's files
stay in your bucket under the job name, MLflow runs stay in your MLflow App, and the role `OpenTSLMSageMakerRole` stays for the next run. Delete them when you no
longer need them.

## License

This example is licensed under the MIT-0 License. See the [LICENSE](../../LICENSE) file of this repository.
`scripts/utils.py` contains Flamingo layers adapted from [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
and [flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch), both under the MIT License. Their notices
are in its docstring.

The pretrained model weights have their own license. See the model card of the checkpoint you use.
