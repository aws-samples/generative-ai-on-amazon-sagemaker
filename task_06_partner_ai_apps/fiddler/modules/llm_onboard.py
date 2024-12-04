import fiddler as fdl
import pandas as pd
import numpy as np
import modules.config as cfg
import time


def create_and_publish_llm(project, model_name: str) -> list[fdl.Model, str]:
    sample_data_df = pd.read_parquet(cfg.PATH_TO_SAMPLE_CHATBOT_CSV)

    sample_data_df['Enrichment Prompt Embedding'] = sample_data_df[
        'Enrichment Prompt Embedding'
    ].apply(lambda x: x.tolist())
    sample_data_df['Enrichment Response Embedding'] = sample_data_df[
        'Enrichment Response Embedding'
    ].apply(lambda x: x.tolist())

    fiddler_backend_enrichments = [
        # prompt enrichment
        fdl.TextEmbedding(
            name='PromptTextEmbedding',
            source_column='user_input',
            column='Enrichment Prompt Embedding',
            n_tags=5,
        ),
        # response enrichment
        fdl.TextEmbedding(
            name='ResponseTextEmbedding',
            source_column='chatbot_response',
            column='Enrichment Response Embedding',
            n_tags=5,
        ),
    ]

    model_spec = fdl.ModelSpec(
        inputs=['user_input', 'chatbot_response'],
        metadata=list(
            sample_data_df.drop(['user_input', 'chatbot_response'], axis=1).columns
        ),
        custom_features=fiddler_backend_enrichments,
    )

    model_task = fdl.ModelTask.LLM

    timestamp_column = 'timestamp'

    llm_application = None
    # Create model
    try:
        llm_application = fdl.Model.from_data(
            source=sample_data_df,
            name=model_name,
            project_id=project.id,
            spec=model_spec,
            task=model_task,
            event_ts_col=timestamp_column,
            max_cardinality=3,
        )
        llm_application.create()
    except fdl.Conflict:
        llm_application = fdl.Model.from_name(
            name=model_name,
            project_id=project.id,
        )

    print(
        f'LLM application registered with id = {llm_application.id} and name = {llm_application.name}'
    )

    segment_definitions = [
        ("Click", "User clicked", "result=='click'"),
        ("No Click", "User did not click", "result=='no_click'"),
        ("Booked", "User Booked", "result=='booked'"),
        ("Liked Answers", "User Liked Answers", "feedback=='like'"),
        ("Disliked Answers", "User Disliked Answers", "feedback=='dislike'")
    ]

    for name, description, definition in segment_definitions:
        try:
            fdl.Segment(
                name=name,
                model_id=llm_application.id,
                description=description,
                definition=definition,
            ).create()
        except fdl.Conflict:
            print(f"Segment '{name}' already exists.")

    custom_metrics = [
        ("Total Cost", "Cost in USD", "sum((prompt_tokens*0.01)+(completion_tokens*0.03))"),
        ("Prompt Token Cost", "Cost in USD", "sum((prompt_tokens*0.01))"),
        ("Response Token Cost", "Cost in USD", "sum((completion_tokens*0.03))"),
    ]

    for name, description, definition in custom_metrics:
        try:
            fdl.CustomMetric(
                name=name,
                model_id=llm_application.id,
                description=description,
                definition=definition,
            ).create()
        except fdl.Conflict:
            print(f"Custom Metric '{name}' already exists.")

    llm_events_df = sample_data_df
    # Timeshifting the timestamp column in the events file so the events are as recent as today
    llm_events_df['timestamp'] = np.linspace(
        int(time.time()) - (5 * 24 * 60 * 60), int(time.time()), num=llm_events_df.shape[0]
    )

    print('Printing sample dataset...')
    print(llm_events_df.head(10).to_markdown())
    return llm_application, llm_application.publish(llm_events_df).id
