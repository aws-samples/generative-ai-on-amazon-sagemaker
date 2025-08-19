import logging
import yaml

import fiddler as fdl
from fiddler.libs.http_client import RequestClient

logger = logging.getLogger(__name__)


def add_chart(project: fdl.Project, model: fdl.Model, unique_id: str, client: RequestClient, chart: dict):
    charts_url = '/v3/charts'
    title = f'[{unique_id}] {chart.get("title")}'
    chart['title'] = title

    for index, query in enumerate(chart['data_source']['queries']):
        version = query.get('version', 'v1')
        query.update(
            {
                'model': {'id': model.id, 'name': model.name},
                'model_name': model.name,
                'version': version,
            }
        )

        baseline_name = query.get('baseline_name')
        if baseline_name:
            baseline = fdl.Baseline.from_name(name=baseline_name, model_id=model.id)
            baseline_id = baseline.id
            query['baseline_id'] = baseline_id
            del query['baseline_name']

        if query.get('metric_type') == 'custom':
            custom_metrics = fdl.CustomMetric.from_name(
                name=query.get('metric'), model_id=model.id
            )
            query['metric'] = custom_metrics.id

        segment = query.get('segment')
        if segment:
            segment = fdl.Segment.from_name(name=segment, model_id=model.id)
            query['segment'] = {}
            query['segment']['id'] = segment.id

        chart['data_source']['queries'][index] = query
    chart['project_id'] = project.id
    client.post(url=charts_url, data=chart)


def add_charts(
    project: fdl.Project,
    model: fdl.Model,
    unique_id: str,
    filename: str,
    fiddler_url: str,
    token: str,
) -> list:
    charts = None
    with open(filename, 'r') as stream:
        try:
            charts = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    errors = []
    if charts and len(charts) <= 0:
        print("no charts found")
        return []
    
    client = RequestClient(
        fiddler_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        },
    )

    for chart in charts.get('charts'):
        try:
            add_chart(project, model, unique_id, client, chart)
        except Exception as exc:
            message = f'Exception {str(exc)} for adding charts'
            logger.error(message)
            errors.append(
                {
                    'chart': 'chart',
                    'status': 'FAILED',
                    'message': message,
                }
            )
            continue

    return errors
