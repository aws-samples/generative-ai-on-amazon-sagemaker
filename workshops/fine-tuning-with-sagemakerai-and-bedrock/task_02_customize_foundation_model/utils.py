import matplotlib.pyplot as plt
from typing import List, Dict
import boto3

def calculate_message_lengths(dataset: List[Dict]) -> List[int]:
    """
    Calculate the length of content/text for each element in the dataset.
    
    Args:
        dataset: List of dictionaries containing messages or text
        
    Returns:
        List of word counts for each element
    """
    try:
        # First try to process as messages format
        return [sum(len(msg["content"].split()) 
                   for msg in element["messages"]) 
                for element in dataset]
    except KeyError:
        # Fallback to direct text/content format
        key = "content" if "content" in dataset[0] else "text"
        return [len(element[key].split()) for element in dataset]

def plot_length_distribution(train_dataset: List[Dict], 
                           validation_dataset: List[Dict],
                           bins: int = 20,
                           figsize: tuple = (10, 6)) -> None:
    """
    Plot the distribution of text lengths from training and validation datasets.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        bins: Number of histogram bins
        figsize: Figure size as (width, height)
    """
    # Calculate lengths for both datasets
    train_lengths = calculate_message_lengths(train_dataset)
    val_lengths = calculate_message_lengths(validation_dataset)
    combined_lengths = train_lengths + val_lengths
    
    # Create and configure the plot
    plt.figure(figsize=figsize)
    plt.hist(combined_lengths, 
            bins=bins, 
            alpha=0.7, 
            color="blue")
    
    # Set labels and title
    plt.xlabel("Prompt Lengths (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Input Lengths")
    
    plt.show()
    

def get_last_job_name(job_name_prefix):
    sagemaker_client = boto3.client('sagemaker')

    matching_jobs = []
    next_token = None

    while True:
        # Prepare the search parameters
        search_params = {
            'Resource': 'TrainingJob',
            'SearchExpression': {
                'Filters': [
                    {
                        'Name': 'TrainingJobName',
                        'Operator': 'Contains',
                        'Value': job_name_prefix
                    },
                    {
                        'Name': 'TrainingJobStatus',
                        'Operator': 'Equals',
                        'Value': "Completed"
                    }
                ]
            },
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending',
            'MaxResults': 100
        }

        # Add NextToken if we have one
        if next_token:
            search_params['NextToken'] = next_token

        # Make the search request
        search_response = sagemaker_client.search(**search_params)

        # Filter and add matching jobs
        matching_jobs.extend([
            job['TrainingJob']['TrainingJobName'] 
            for job in search_response['Results']
            if job['TrainingJob']['TrainingJobName'].startswith(job_name_prefix)
        ])

        # Check if we have more results to fetch
        next_token = search_response.get('NextToken')
        if not next_token or matching_jobs:  # Stop if we found at least one match or no more results
            break

    if not matching_jobs:
        raise ValueError(f"No completed training jobs found starting with prefix '{job_name_prefix}'")

    return matching_jobs[0]